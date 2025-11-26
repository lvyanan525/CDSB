import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset

class I2I_Dataset(Dataset):
    def __init__(self, image_dir, condition_dir, resolution=256, gray=False, exclude=False, debug_num=None):
        self.image_dir = image_dir
        self.condition_dir = condition_dir
        self.resolution = resolution
        self.gray = gray
        
        # 检查目录是否存在
        if not os.path.exists(image_dir):
            raise ValueError(f"图像目录不存在: {image_dir}")
        if not os.path.exists(condition_dir):
            raise ValueError(f"条件图像目录不存在: {condition_dir}")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp'))])
        self.condition_files = sorted([f for f in os.listdir(condition_dir) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp'))])
        if exclude:
            exclude_image_test = image_dir.replace('17k','test')
            exclude_condition_test = condition_dir.replace('17k','test')
            exclude_image_val = image_dir.replace('17k','val')
            exclude_condition_val = condition_dir.replace('17k','val')
            exclude_image_files = sorted([f for f in os.listdir(exclude_image_test) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp'))]
                                         +[f for f in os.listdir(exclude_image_val) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp'))])
            exclude_condition_files = sorted([f for f in os.listdir(exclude_condition_test) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp'))]
                                             +[f for f in os.listdir(exclude_condition_val) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp'))])
            self.image_files = [f for f in self.image_files if f not in exclude_image_files]
            self.condition_files = [f for f in self.condition_files if f not in exclude_condition_files]

        if debug_num is not None:
            self.image_files = self.image_files[:debug_num]
            self.condition_files = self.condition_files[:debug_num]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        condition_path = os.path.join(self.condition_dir, self.condition_files[idx])
        
        try:
            # 加载图像
            image = Image.open(image_path)
            if self.gray:
                image = image.convert("L")
            else:
                image = image.convert("RGB")
            image = self.transform(image)
            condition = Image.open(condition_path)
            if self.gray:
                condition = condition.convert("L")
            else:
                condition = condition.convert("RGB")
            condition = self.transform(condition)
        except Exception as e:
            raise RuntimeError(f"加载图像文件失败 {self.image_files[idx]}: {str(e)}")
        
        return {
            "pixel_values": image,
            "file_name": self.image_files[idx],
            "condition": condition
        }

if __name__ == "__main__":
    # 创建数据集实例，使用tokenizer
    dataset = I2I_Dataset(
        image_dir="dataset/lm2em/17k/B", 
        condition_dir="dataset/lm2em/17k/A", 
        resolution=256,
        exclude=True
    )
    print(dataset[0])
    dataset = I2I_Dataset(image_dir="dataset/control_net_dataset/images", condition_dir="dataset/control_net_dataset/conditioning_images", prompt=None, tokenizer=None, resolution=256)
    print(dataset[0])