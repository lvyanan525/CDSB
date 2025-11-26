import os
import torch
import torchvision
import torchmetrics
import lightning as L
from torch.utils.data import DataLoader, Dataset
from emsb.runner import emsb_runner
from dataset.customdataset import I2I_Dataset
import argparse
import json
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import numpy as np

class InferenceDataset(Dataset):
    """只处理输入图像的数据集，用于推理"""
    def __init__(self, input_dir, resolution=512):
        self.input_dir = input_dir
        self.resolution = resolution
        
        # 检查目录是否存在
        if not os.path.exists(input_dir):
            raise ValueError(f"输入图像目录不存在: {input_dir}")
        
        # 图像预处理 - 保持原始尺寸
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp'))])
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        
        try:
            # 加载输入图像
            input_image = Image.open(input_path).convert("RGB")
            input_image = self.transform(input_image)
        except Exception as e:
            raise RuntimeError(f"加载图像文件失败 {self.input_files[idx]}: {str(e)}")
        
        return {
            "condition": input_image,
            "file_name": self.input_files[idx]
        }

class PatchProcessor:
    """图像分块处理器转接器"""
    def __init__(self, model, patch_size=256, inference_batch_size=16, device='cuda', memory_efficient=True):
        self.model = model
        self.patch_size = patch_size
        self.inference_batch_size = inference_batch_size
        self.device = device
        self.memory_efficient = memory_efficient
    
    def calculate_optimal_dataloader_batch_size(self, image_size=512):
        """自动计算最优的dataloader batch size"""
        # 计算每张图像会产生多少个patches
        num_patches_per_image = (image_size // self.patch_size) ** 2
        if image_size % self.patch_size != 0:
            num_patches_per_image += 2  # 边界patches
        
        # 基于内存效率计算最优dataloader batch size
        if self.memory_efficient:
            # 保守策略：确保不会占用过多GPU内存
            optimal_batch_size = max(1, min(4, 16 // num_patches_per_image))
        else:
            # 激进策略：尽可能利用GPU内存
            optimal_batch_size = max(1, min(8, 32 // num_patches_per_image))
        
        return optimal_batch_size
    
    def split_image_to_patches(self, image):
        """将图像分割成patches"""
        # image: [C, H, W]
        C, H, W = image.shape
        
        patches = []
        positions = []
        
        for i in range(0, H, self.patch_size):
            for j in range(0, W, self.patch_size):
                # 确保patch不超出边界
                end_i = min(i + self.patch_size, H)
                end_j = min(j + self.patch_size, W)
                
                # 如果patch不足patch_size，进行padding
                if end_i - i < self.patch_size or end_j - j < self.patch_size:
                    patch = torch.zeros(C, self.patch_size, self.patch_size, device=image.device)
                    patch[:, :end_i-i, :end_j-j] = image[:, i:end_i, j:end_j]
                else:
                    patch = image[:, i:end_i, j:end_j]
                
                patches.append(patch)
                positions.append((i, j, end_i, end_j))
        
        return patches, positions
    
    def merge_patches_to_image(self, patches, positions, original_shape):
        """将patches合并回原始图像"""
        C, H, W = original_shape
        merged_image = torch.zeros(C, H, W, device=patches[0].device)
        
        for patch, (i, j, end_i, end_j) in zip(patches, positions):
            # 只取有效区域
            patch_h, patch_w = end_i - i, end_j - j
            merged_image[:, i:end_i, j:end_j] = patch[:, :patch_h, :patch_w]
        
        return merged_image
    
    def process_image(self, image):
        """处理单张图像"""
        # 分割图像
        patches, positions = self.split_image_to_patches(image)
        
        # 批量处理patches
        processed_patches = []
        for i in range(0, len(patches), self.inference_batch_size):
            batch_patches = patches[i:i + self.inference_batch_size]
            batch_tensor = torch.stack(batch_patches).to(self.device)
            
            # 推理
            with torch.no_grad():
                batch_outputs = self.model.inference(batch_tensor, nfe=20)
            
            processed_patches.extend(batch_outputs)
        
        # 合并结果
        merged_image = self.merge_patches_to_image(processed_patches, positions, image.shape)
        
        return merged_image
    
    def process_batch(self, batch):
        """处理一个batch的图像"""
        input_imgs = batch['condition']
        filenames = batch['file_name']
        
        processed_images = []
        for i in range(len(filenames)):
            # 处理单张图像
            processed_img = self.process_image(input_imgs[i])
            processed_images.append(processed_img)
        
        return {
            'processed_images': processed_images,
            'file_name': filenames
        }
    
    def create_dataloader(self, dataset, num_workers=4):
        """自动创建最优的dataloader"""
        optimal_batch_size = self.calculate_optimal_dataloader_batch_size()
        print(f"自动计算的最优dataloader batch size: {optimal_batch_size}")
        
        return DataLoader(
            dataset,
            batch_size=optimal_batch_size,
            num_workers=num_workers,
            shuffle=False
        )

def load_checkpoint(checkpoint_path, model_class, device='cuda'):
    """使用Lightning官方方法加载checkpoint并设置设备"""
    model = model_class.load_from_checkpoint(checkpoint_path, map_location='cpu')
    model = model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="/home/lvyn/mira_datacenter/sequence_512/optical_seq_512", help='输入图像文件夹')
    parser.add_argument('--gt_dir', default=None, help='GT图像文件夹（可选）')
    parser.add_argument('--emsb_ckpt', default="logs/emsb_stl_0.1blur/checkpoints/emsb-step=9072-val_mse=0.0325.ckpt", help='EMSB checkpoint路径')
    parser.add_argument('--nfe', type=int, default=20, help='NFE步数')
    parser.add_argument('--save_dir', default='inference_results512', help='推理结果保存目录')
    parser.add_argument('--inference_batch_size', type=int, default=4, help='推理批次大小（控制输入网络的batch size）')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='使用的GPU设备')
    parser.add_argument('--patch_size', type=int, default=512, help='分块大小')
    parser.add_argument('--memory_efficient', action='store_true', help='使用内存保守策略')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置设备
    device = f'cuda:{args.devices[0]}' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载EMSB模型...")
    emsb_model = load_checkpoint(args.emsb_ckpt, emsb_runner, device)
    # 透传nfe用于inference路径
    setattr(emsb_model.hparams, 'nfe', args.nfe)
    
    # 创建分块处理器
    patch_processor = PatchProcessor(
        model=emsb_model,
        patch_size=args.patch_size,
        inference_batch_size=args.inference_batch_size,
        device=device,
        memory_efficient=args.memory_efficient
    )
    
    # 根据是否有GT选择数据集
    if args.gt_dir:
        # 有GT时使用原始I2I_Dataset
        dataset = I2I_Dataset(
            image_dir=args.gt_dir,
            condition_dir=args.input_dir
        )
    else:
        # 没有GT时使用新的InferenceDataset
        dataset = InferenceDataset(input_dir=args.input_dir)
    
    dataloader = patch_processor.create_dataloader(dataset, num_workers=args.num_workers)
    
    # 初始化指标（仅在提供GT时使用）
    if args.gt_dir:
        metrics = torchmetrics.MetricCollection([
            torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0),
            torchmetrics.MeanSquaredError()
        ])
    
    results = []
    
    print(f"开始处理 {len(dataset)} 张图像...")
    print(f"推理结果将保存到: {args.save_dir}")
    print(f"分块大小: {args.patch_size}x{args.patch_size}")
    print(f"推理批次大小: {args.inference_batch_size}")
    print(f"数据加载器批次大小: {patch_processor.calculate_optimal_dataloader_batch_size()}")
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc="处理图像", unit="batch")
    
    for batch_idx, batch in enumerate(pbar):
        # 将数据移到设备上
        batch['condition'] = batch['condition'].to(device)
        if args.gt_dir:
            batch['pixel_values'] = batch['pixel_values'].to(device)
        
        # 使用分块处理器处理batch
        processed_batch = patch_processor.process_batch(batch)
        
        # 处理每张图像
        for i, (processed_img, filename) in enumerate(zip(processed_batch['processed_images'], processed_batch['file_name'])):
            # 保存推理结果
            output_path = os.path.join(args.save_dir, f"{os.path.splitext(filename)[0]}_pred.png")
            # 将张量从[-1,1]范围转换到[0,1]范围并保存
            save_image(processed_img.unsqueeze(0) / 2 + 0.5, output_path, normalize=False)
            
            # 如果有GT，计算指标
            if args.gt_dir:
                gt_img = batch['pixel_values'][i:i+1]
                emsb_metrics = metrics(processed_img.unsqueeze(0).detach().cpu() / 2 + 0.5, gt_img.detach().cpu() / 2 + 0.5)
                
                # 记录结果
                result = {
                    'filename': filename,
                    'ssim': emsb_metrics['StructuralSimilarityIndexMeasure'].item(),
                    'mse': emsb_metrics['MeanSquaredError'].item()
                }
                results.append(result)
        
        # 更新进度条描述
        pbar.set_description(f"处理图像 (已处理: {batch_idx + 1}/{len(dataloader)} 批次)")
    
    # 关闭进度条
    pbar.close()
    
    # 如果有GT，保存指标结果
    if args.gt_dir and results:
        metrics_file = os.path.join(args.save_dir, 'metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 计算平均指标
        avg_ssim = sum(r['ssim'] for r in results) / len(results)
        avg_mse = sum(r['mse'] for r in results) / len(results)
        
        print(f"\n推理完成！")
        print(f"平均SSIM: {avg_ssim:.4f}")
        print(f"平均MSE: {avg_mse:.4f}")
        print(f"指标详情已保存到: {metrics_file}")
    else:
        print(f"\n推理完成！共处理 {len(dataset)} 张图像")
    
    print(f"推理结果已保存到: {args.save_dir}")

if __name__ == '__main__':
    main() 