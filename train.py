import argparse
import os

from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.plugins.io import AsyncCheckpointIO
from lightning.pytorch.loggers import TensorBoardLogger
from emsb.runner import emsb_runner 
from dataset.customdataset import I2I_Dataset

def get_args():
    parser = argparse.ArgumentParser(description="训练 emsb 模型的脚本")
    
    # --- 日志和模型保存目录 ---
    parser.add_argument('--log_dir', type=str, default='logs/', help='日志和模型保存目录')
    parser.add_argument('--name', type=str, default='emsb_logs', help='日志和模型保存目录')
    # --- 模型超参数 ---
    parser.add_argument('--n_timestep', type=int, default=1000, help='扩散模型的总步数')
    parser.add_argument('--beta_max', type=float, default=0.3, help='Beta schedule 的最大值')
    parser.add_argument('--t0', type=float, default=1e-4, help='时间步的起始值')
    parser.add_argument('--T', type=float, default=1.0, help='时间步的结束值')
    parser.add_argument('--use_fp16', action='store_true', help='是否使用半精度(float16)')
    parser.add_argument('--cond_x1', action='store_true', help='是否开启条件控制')
    parser.add_argument('--ema', type=float, default=0.99, help='Exponential Moving Average 的衰减率')
    parser.add_argument('--ot_ode', action='store_true', help='是否使用 OT-ODE 进行采样')
    
    # --- 优化器和学习率 ---
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--lr_brightness', type=float, default=5e-5, help='亮度调整网络的学习率')
    parser.add_argument('--scheduler_type', type=str, default='warmup_cosine', help='调度器类型')

    # --- 训练过程参数 ---
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='梯度累积的批次数 (microbatching). 有效批量大小 = batch_size * num_devices * accumulate_grad_batches')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--max_epochs', type=int, default=-1, help='最大训练轮数, -1 表示不限制')
    parser.add_argument('--max_steps', type=int, default=-1, help='最大训练步数, -1 表示不限制')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的进程数')
    parser.add_argument('--devices', type=int, nargs='+', default=[2], help='使用的设备')

    parser.add_argument('--val_every_n_batches', type=int, default=2000, help='每 N 个 batch (optimizer steps) 运行一次验证')
    parser.add_argument('--save_every_n_steps', type=int, default=10000, help='每 N 个 step 保存一次固定检查点')

    parser.add_argument('--brightness_net', action='store_true', help='是否使用亮度调整网络')
    parser.add_argument('--brightness_type', type=str, default='stl',choices=['naive','stl','l2','vae'], help='亮度调整类型')
    parser.add_argument('--brightness_channel', type=int, default=1, help='亮度通道数')
    parser.add_argument('--blur_loss', type=float, default=0, help='模糊损失的权重')

    # --- 加载/恢复 ---
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='从指定的检查点文件恢复训练')


    args = parser.parse_args()
    return args

def main(opt):
    """主训练函数"""
    seed_everything(42)
    # 1. 设置模型
    model = emsb_runner(opt)

    # 2. 设置数据加载器 (DataLoaders)
    # 请务必替换为您的真实数据集
    train_dataset = I2I_Dataset(image_dir="dataset/lm2em2/train/B", condition_dir="dataset/lm2em2/train/A")
    val_dataset = I2I_Dataset(image_dir="dataset/lm2em2/val/B", condition_dir="dataset/lm2em2/val/A")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    async_checkpoint_io = AsyncCheckpointIO()
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(opt.log_dir, opt.name, "checkpoints"),
        filename='emsb-step={step:02d}-val_mse={VAL/val_mse:.4f}',
        monitor='VAL/val_ssim',
        mode='max',
        save_top_k=3,
        auto_insert_metric_name=False
    )

        # 回调 B: 每隔 N 步固定保存一个检查点
    step_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(opt.log_dir, opt.name, "checkpoints"),
        filename='emsb-step-{step}',
        every_n_train_steps=opt.save_every_n_steps,
        save_top_k=-1, # -1 表示保存所有符合条件的检查点
        save_last=True # 同时保存一个 last.ckpt 的软链接
    )


    # b. 学习率监视器
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 4. 设置日志记录器 (Logger)
    logger = TensorBoardLogger(opt.log_dir, name=opt.name)

    # 5. 初始化并配置 Trainer
    trainer = Trainer(
        devices=opt.devices,
        max_epochs=opt.max_epochs,
        max_steps=opt.max_steps,
        logger=logger,
        callbacks=[checkpoint_callback, step_checkpoint_callback, lr_monitor],
        precision='16-mixed' if opt.use_fp16 else 32, # 混合精度训练
        log_every_n_steps=4,
        val_check_interval=opt.val_every_n_batches, # 每 n 步验证一次
        check_val_every_n_epoch=None,
        accumulate_grad_batches=opt.accumulate_grad_batches,
        plugins=[async_checkpoint_io]
    )

    # 6. 开始训练
    #    - `ckpt_path`: 如果提供了这个参数，Trainer会自动加载检查点并恢复训练
    print(f"开始训练... 日志和模型将保存在: {opt.log_dir}")
    if opt.resume_from_checkpoint:
        print(f"从检查点恢复训练: {opt.resume_from_checkpoint}")

    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader,
        ckpt_path=opt.resume_from_checkpoint
    )


if __name__ == '__main__':
    args = get_args()
    # 确保日志目录存在
    os.makedirs(args.log_dir, exist_ok=True)
    main(args)

