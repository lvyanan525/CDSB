import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

def augment_tensor_neg1_1(tensor: torch.Tensor) -> torch.Tensor:
    """
    对一个[-1, 1]范围的PyTorch Tensor进行带蒙版的亮度/对比度/伽马增强。
    这个函数是为 (b, c, h, w) 的4D张量设计的，并且会在batch内进行独立的随机增强。

    Args:
        tensor (torch.Tensor): 输入的张量，shape=(b, c, h, w), range=[-1, 1]。

    Returns:
        torch.Tensor: 增强后的张量，shape和range与输入相同。
    """
    # 获取batch size和device
    b, c, h, w = tensor.shape
    device = tensor.device

    # --- 1. 创建蒙版 ---
    # 截断值现在是 -1.0 和 1.0
    # 对多通道，只要有一个通道是截断值，就屏蔽该像素
    mask = torch.any((tensor <= -1.0) | (tensor >= 1.0), dim=1, keepdim=True)
    # (b, 1, h, w)
    
    # 将蒙版扩展到与输入张量相同的通道数
    mask = mask.expand_as(tensor) # (b, c, h, w)
    
    # 反向蒙版，方便计算
    non_mask = ~mask

    # --- 2. 随机应用一系列增强 ---
    augmented_tensor = tensor.clone()

    # 对batch中的每个样本应用不同的增强参数
    
    # a. 对比度 (alpha)
    if torch.rand(1).item() > 0.5:
        # 为batch中的每个图像生成一个独立的alpha值
        alpha = torch.rand(b, device=device) * 1 + 0.5  # e.g., range [0.65, 1.35]
        alpha = alpha.view(b, 1, 1, 1) # Reshape for broadcasting -> (b, 1, 1, 1)
        
        # 对比度调整是以0为中心的，在[-1, 1]范围内直接乘法即可
        augmented_tensor[non_mask] = augmented_tensor[non_mask] * alpha.expand_as(augmented_tensor)[non_mask]
    
    # b. 亮度 (beta)
    if torch.rand(1).item() >0.5:
        beta = (torch.rand(b, device=device) - 0.5) * 0.4 # e.g., range [-0.2, 0.2]
        beta = beta.view(b, 1, 1, 1)
        
        # 亮度是加法，直接应用 
        augmented_tensor[non_mask] = augmented_tensor[non_mask] + beta.expand_as(augmented_tensor)[non_mask]

    # c. 伽马校正 (gamma)
    if True:
        gamma = torch.pow(2, torch.rand(b, device=device) * 2 - 1) # e.g., range [0.5, 2]
        gamma = gamma.view(b, 1, 1, 1)
        
        # -- 这里是关键：范围映射 --
        # 1. 从[-1, 1]映射到[0, 1]
        val_0_1 = (augmented_tensor[non_mask] + 1.0) / 2.0
        val_0_1 = torch.clamp(val_0_1, 0, 1)
        # 2. 在[0, 1]上应用伽马
        epsilon = 1e-7
        val_0_1_gamma = torch.pow(val_0_1 + epsilon, gamma.expand_as(augmented_tensor)[non_mask])

        # 3. 映射回[-1, 1]
        val_neg1_1_gamma = val_0_1_gamma * 2.0 - 1.0
        
        augmented_tensor[non_mask] = val_neg1_1_gamma
        
    # --- 3. 后处理 ---
    # 将所有像素值裁剪回 [-1, 1] 范围
    augmented_tensor = torch.clamp(augmented_tensor, -1.0, 1.0)
    
    return augmented_tensor

def bright_norm(o, params=None, net_f=None, type='naive'):
    if type == 'naive':
        x = (o - o.mean(dim=(1,2,3), keepdim=True)) / (o.std(dim=(1,2,3), keepdim=True) + 1e-6)
    elif type == 'vae':
        x = net_f(o).latent_dist.sample() * 0.18215
    else:
        if params is None:
            params = torch.cat([o.mean(dim=(1,2,3)).unsqueeze(1), o.std(dim=(1,2,3)).unsqueeze(1)], dim=1)
        x = net_f(o, params)
    return x

def bright_recon(x, params=None, net_b=None, type='naive'):
    if type == 'naive':
        x = x * 0.3
    elif type == 'vae':
        x = net_b(x / 0.18215).sample
    else:
        if params is None:
            out_mean = torch.tensor([[0.087878]], device=x.device)
            out_std = torch.tensor([[0.309309]], device=x.device)
            params = torch.cat([out_mean, out_std], dim=1)
            params = params.repeat(x.shape[0], 1)
        x = net_b(x, params)
    return x

def bright_loss(net_f, net_b, o0, o1, brightness_type='naive'):
    if brightness_type == 'naive':
        x0 = bright_norm(o0, net_f, type=brightness_type)
        x1 = bright_norm(o1, net_f, type=brightness_type)
        loss_norm = 0
        loss_recon = 0
        loss_var = 0
        loss_brightness = 0
    elif brightness_type == 'vae':
        x0 = net_f(o0).latent_dist.sample() * 0.18215
        x1 = net_f(o1).latent_dist.sample() * 0.18215
        loss_norm = 0
        loss_recon = 0
        loss_var = 0
        loss_brightness = 0
    else:
        img_aug_0 = augment_tensor_neg1_1(o0)
        img_aug_1 = augment_tensor_neg1_1(o1)

        # # x0亮度和标准差
        brightness_params_o = torch.cat([o0.mean(dim=(1,2,3)).unsqueeze(1), o0.std(dim=(1,2,3)).unsqueeze(1)], dim=1)
        brightness_params_aug_o = torch.cat([img_aug_0.mean(dim=(1,2,3)).unsqueeze(1), img_aug_0.std(dim=(1,2,3)).unsqueeze(1)], dim=1)
        brightness_params_i = torch.cat([o1.mean(dim=(1,2,3)).unsqueeze(1), o1.std(dim=(1,2,3)).unsqueeze(1)], dim=1)
        brightness_params_aug_i = torch.cat([img_aug_1.mean(dim=(1,2,3)).unsqueeze(1), img_aug_1.std(dim=(1,2,3)).unsqueeze(1)], dim=1)
        
        x0 = bright_norm(o0, brightness_params_o, net_f, type=brightness_type)
        x1 = bright_norm(o1, brightness_params_i, net_f, type=brightness_type)
        x0_aug = bright_norm(img_aug_0, brightness_params_aug_o, net_f, type=brightness_type)
        x1_aug = bright_norm(img_aug_1, brightness_params_aug_i, net_f, type=brightness_type)
        
        # 反向亮度变换
        o0_bar = bright_recon(x0, brightness_params_o, net_b, type=brightness_type)
        o1_bar = bright_recon(x1, brightness_params_i, net_b, type=brightness_type)

        # x 和 x_aug应该是一样的
        loss_norm = F.l1_loss(x0, x0_aug) + F.l1_loss(x1, x1_aug)
        loss_recon = F.l1_loss(o0_bar, o0) + F.l1_loss(o1_bar, o1)
        if brightness_type == 'l2':
            target_var = torch.ones_like(x0.var(dim=(2,3))) * 1.0
            loss_var = F.mse_loss(x0.var(dim=(2,3)), target_var) + \
               F.mse_loss(x1.var(dim=(2,3)), target_var)
            loss_brightness = loss_recon + 0.1 * loss_norm + 0.1 * loss_var
        elif brightness_type == 'stl':
            loss_var = torch.mean(1 / (x0.var(dim=(2,3)) + 1e-6) + (x0.var(dim=(2,3)) + 1e-6) + \
                                1 / (x1.var(dim=(2,3)) + 1e-6) + (x1.var(dim=(2,3)) + 1e-6))
            loss_brightness = loss_recon + 0.1 * loss_norm + 0.01 * loss_var
        else:
            raise ValueError(f"Invalid brightness_type: {brightness_type}")
    return x0, x1, loss_recon, loss_norm, loss_var, loss_brightness

# --- 使用示例 ---
if __name__ == '__main__':
    # 从指定路径读取图像
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 设置图像变换
    transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 将[0,1]转换为[-1,1]
    ])

    # 检查路径是否存在
    dataset_path = "dataset/lm2em/test"
    if not os.path.exists(dataset_path):
        print(f"错误：路径 {dataset_path} 不存在")
        exit(1)

    # 加载数据集
    try:
        dataset = ImageFolder(root=dataset_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # 获取一个batch的图像
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx == 0:  # 只处理第一个batch
                dummy_batch = images.to(device)
                break
        else:
            print("数据集为空")
            exit(1)
            
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        exit(1)

    # 在第一张图中手动制造一些截断区域
    dummy_batch[0, :, 10:30, 10:30] = -1.0
    dummy_batch[0, :, 40:50, 40:50] = 1.0
    
    print("Original tensor min/max:", dummy_batch.min().item(), dummy_batch.max().item())

    # 进行增强
    augmented_batch = augment_tensor_neg1_1(dummy_batch)
    
    print("Augmented tensor min/max:", augmented_batch.min().item(), augmented_batch.max().item())
    
    # 检查截断区域是否被改变
    original_clipped_mask = (dummy_batch == -1.0) | (dummy_batch == 1.0)
    augmented_clipped_values = augmented_batch[original_clipped_mask]
    
    if torch.all(dummy_batch[original_clipped_mask] == augmented_clipped_values):
        print("SUCCESS: Clipped areas were preserved.")
    else:
        print("ERROR: Clipped areas were altered.")
        
    # 可视化检查 (需要matplotlib和torchvision)
    try:
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        # 将[-1, 1]的张量转换为[0, 1]以供显示
        def to_displayable(tensor):
            return (tensor + 1.0) / 2.0

        grid_original = make_grid(to_displayable(dummy_batch.cpu()), nrow=4)
        grid_augmented = make_grid(to_displayable(augmented_batch.cpu()), nrow=4)

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        axs[0].imshow(grid_original.permute(1, 2, 0))
        axs[0].set_title('Original Batch')
        axs[0].axis('off')

        axs[1].imshow(grid_augmented.permute(1, 2, 0))
        axs[1].set_title('Augmented Batch')
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("brightnessloss_augmented_result.png")
        print("图像已保存为 brightnessloss_augmented_result.png")

    except ImportError:
        print("\nPlease install matplotlib and torchvision (`pip install matplotlib torchvision`) to visualize the results.")