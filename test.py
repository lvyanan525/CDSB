import os
import argparse
import torch
import lightning as L
import json
from torch.utils.data import DataLoader
from emsb.runner import emsb_runner
from dataset.customdataset import I2I_Dataset

def main():
    parser = argparse.ArgumentParser(description='测试EMSB模型')
    
    # 数据和模型路径
    parser.add_argument('--log_dir', type=str, default='logs/', help='日志和模型保存目录')
    parser.add_argument('--name', type=str, default='emsb_naive', help='实验名称')
    parser.add_argument('--checkpoint', type=str, default='emsb-step=6048-val_mse=0.0347.ckpt', help='checkpoint文件名（默认: last.ckpt）')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='使用的GPU设备')
    parser.add_argument('--nfe', type=int, default=20, help='评估步数')
    parser.add_argument('--eval_root', type=str, default=None, help='统一评估结果根目录；若提供，将落盘到 eval_root/step_{nfe}/')
    args = parser.parse_args()
    
    # 创建测试数据集
    test_dataset = I2I_Dataset(
        image_dir="dataset/lm2em2/test/B", condition_dir="dataset/lm2em2/test/A"
    )
    
    # 创建数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    # 构建checkpoint完整路径
    checkpoint_path = os.path.join(args.log_dir, args.name, 'checkpoints', args.checkpoint)
    print(f"加载模型: {checkpoint_path}")
    
    # 加载模型，同时传递必要的超参数以覆盖checkpoint中的值
    model = emsb_runner.load_from_checkpoint(
        checkpoint_path,    
        map_location='cpu',
        log_dir=args.log_dir,
        name=args.name
    )
    setattr(model.hparams, 'nfe', args.nfe)
    if args.eval_root is not None:
        setattr(model.hparams, 'eval_save_dir', os.path.join(args.eval_root, f"step_{args.nfe}"))

    # 创建trainer
    trainer = L.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # accelerator='cpu',
        devices=args.devices,
        logger=False,  # 测试时不需要logger
        enable_checkpointing=False
    )
    # 运行测试
    save_dir = getattr(model.hparams, 'eval_save_dir', os.path.join(args.log_dir, args.name, "test"))
    print(f"开始测试，共 {len(test_dataset)} 张图像...")
    print(f"测试步数为：{model.hparams.nfe}")
    print(f"结果将保存到: {save_dir}")
    trainer.test(model, test_dataloader)
    print("测试完成！")

if __name__ == "__main__":
    main() 