import os
from typing import Optional
import torch
import numpy as np
import torchvision
import torchmetrics
import lightning as L
from emsb.network import Image256Net
from emsb.FiLMLayer import BrightNormNet
from timm.utils import ModelEmaV2
from diffusers import DDPMScheduler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR
import torch.nn.functional as F

class refiner_runner(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)

        # 构建diffusion
        self.scheduler = DDPMScheduler(
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        self.scheduler.set_timesteps(self.hparams.n_timestep)


        noise_levels = torch.linspace(self.hparams.t0, self.hparams.T, self.hparams.n_timestep) * self.hparams.n_timestep
        # 构建网络和ema
        self.net = Image256Net(noise_levels=noise_levels, use_fp16=self.hparams.use_fp16, cond=self.hparams.cond_x1)
        
        # 初始化亮度调整网络（如果需要）
        if self.hparams.brightness_net:
            self.brightness_net_f = BrightNormNet(brightness_param_dim=2, input_channels=1)
            self.brightness_net_b = BrightNormNet(brightness_param_dim=2, input_channels=1)
            
        self.ema = ModelEmaV2(self.net, decay=self.hparams.ema)

        # 构建metrics
        self.validation_outputs = []
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0),
            # MSE
            torchmetrics.MeanSquaredError()
        ])
        self.org_metrics = torchmetrics.MetricCollection([
            torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0),
            # MSE
            torchmetrics.MeanSquaredError()
        ])
            
    def training_step(self, batch, batch_idx):
        opt = self.hparams
        o0 = batch['pixel_values']
        if opt.brightness_net:
            def norm_loss(x):
                return F.l1_loss(x.mean(), torch.tensor([0.0], device=self.device))
            
            # x0亮度和标准差
            brightness_params_o = torch.cat([o0.mean(dim=(1,2,3)).unsqueeze(1), o0.std(dim=(1,2,3)).unsqueeze(1)], dim=1)
            # 正向亮度变换
            x0 = self.brightness_net_f(o0, brightness_params_o)
            # 反向亮度变换，能将x0还原为o0
            o0_bar = self.brightness_net_b(x0, brightness_params_o)
            # x0，x1的均值要在0附近
            loss_norm = norm_loss(x0)
            loss_recon = F.l1_loss(o0_bar, o0)
            # 计算亮度变换损失
            loss_brightness = loss_recon + 0.01 * loss_norm
        else:
            x0 = o0
            loss_brightness = 0

        noise = torch.randn_like(x0)
        timesteps = torch.randint(0, self.hparams.edit_timestep, (x0.shape[0],), device=self.device)
        self.scheduler.set_timesteps(self.hparams.n_timestep, device=self.device)
        noisy_image = self.scheduler.add_noise(x0, noise, timesteps)

        # 模型预测
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(x0, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")
        
        pred_noise = self.net(noisy_image, timesteps)
            
        loss = F.mse_loss(pred_noise, target) + loss_brightness

        self.log('TRAIN/train_loss', loss, prog_bar=True)
        if opt.brightness_net:
            self.log('TRAIN/loss_norm', loss_norm, prog_bar=True)
            self.log('TRAIN/loss_recon', loss_recon, prog_bar=True)
            self.log('TRAIN/brightness_loss', loss_brightness, prog_bar=True)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        self.ema.update(self.net)
       
    def configure_optimizers(self):
            """专业推荐配置：AdamW + 线性Warmup + 余弦退火"""
            opt = self.hparams
            
            # 收集所有需要优化的参数
            parameters = list(self.net.parameters())
            if opt.brightness_net:
                parameters += list(self.brightness_net_f.parameters())
                parameters += list(self.brightness_net_b.parameters())
            
            optimizer = torch.optim.AdamW(
                parameters, 
                lr=opt.lr
            )
            
            scheduler_type = getattr(self.hparams, 'scheduler_type', 'warmup_cosine')

            if scheduler_type == "constant":
                print(f"使用 'constant' 调度器，学习率固定为: {opt.lr}")
                # LambdaLR 乘以一个恒为1的因子，等效于不改变学习率
                scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
            elif scheduler_type == "warmup_cosine":    
                # 1. 计算总步数和warmup步数
                if self.trainer.max_steps > 0:
                    total_steps = self.trainer.max_steps
                else:
                    total_steps = self.trainer.estimated_stepping_batches

                # print(f"DEBUG: Calculated total_steps = {total_steps}")
                # 假设 warmup 占总步数的 2% (这是一个常见的比例)
                warmup_steps = int(total_steps * 0.02) 

                # 2. 定义两个调度器
                # 线性预热调度器
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=1e-4, # 从一个很小的学习率开始
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                
                # 余弦退火调度器
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps - warmup_steps, # 剩下的步数用于余弦退火
                    eta_min=1e-6
                )
                
                # 3. 使用 SequentialLR 将它们串联起来
                # milestones 指定了在第几步切换到下一个调度器
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
                # 输出关键步数节点信息，便于调试和学习率调度可视化
                print(f"关键步数节点（milestones）: warmup_steps = {warmup_steps}, total_steps = {total_steps}")
            else:
                raise ValueError(f"不支持的调度器类型: {scheduler_type}")
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        bs = batch['pixel_values'].shape[0]
        o0 = batch['pixel_values']
        o1 = batch['condition']
        
        # 如果使用亮度调整网络，需要先转换输入
        if self.hparams.brightness_net:
            brightness_params_i = torch.cat([o1.mean(dim=(1,2,3)).unsqueeze(1), o1.std(dim=(1,2,3)).unsqueeze(1)], dim=1)
            x1 = self.brightness_net_f(o1, brightness_params_i)
        else:
            x1 = o1
        
        pred, noisy_image = self.sdedit_inference(
            x1,
            add_noise_timestep=self.hparams.edit_timestep, # 加 edit_timestep 步的噪声
            denoising_rate=0.1  # Validation: 使用 20 步进行粗略去噪，以节省时间
        )

        # 如果使用亮度调整网络，需要将结果转换回原始亮度空间
        if self.hparams.brightness_net:
            pred = pred.to(self.device)
            out_mean = torch.tensor([[0.087878]], device=self.device)
            out_std = torch.tensor([[0.309309]], device=self.device)
            brightness_params_o = torch.cat([out_mean, out_std], dim=1)
            brightness_params_o = brightness_params_o.repeat(bs, 1)
            pred = self.brightness_net_b(pred, brightness_params_o)

        if batch_idx == 0 and len(self.validation_outputs) == 0:
             self.validation_outputs = [
                 o1[0].detach().cpu(), 
                 noisy_image[0].detach().cpu(),
                 pred[0].detach().cpu(), 
                 o0[0].detach().cpu()
            ]
        self.org_metrics.update(o0.detach().cpu() / 2 + 0.5, o1.detach().cpu() / 2 + 0.5)
        self.metrics.update(pred.detach().cpu() / 2 + 0.5, o0.detach().cpu() / 2 + 0.5)

    @torch.no_grad()
    def sdedit_inference(
        self,
        input_image: torch.Tensor,
        add_noise_timestep: int = None,
        denoising_rate: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        修正后的 SDEdit 推理函数
        1. 对输入图像精确地加入 `add_noise_timestep` 步对应的噪声。
        2. 使用 `denoising_steps` 步从加噪状态完全去噪到清晰图像。

        Args:
            input_image (torch.Tensor): 输入图像 [B, C, H, W]
            add_noise_timestep (int): 要添加噪声的目标步数 (例如 edit_timestep)，对应于训练时的 t。
            denoising_steps (int): 用于去噪的实际步数 (NFE)。
            generator (Optional[torch.Generator]): 随机数生成器。
        """
        if not add_noise_timestep:
            add_noise_timestep = self.hparams.edit_timestep

        # 1. 安全检查：确保加噪不超过训练
        max_train_t = self.hparams.edit_timestep - 1
        start_t = add_noise_timestep -1
        assert start_t <= max_train_t, f"add_noise_timestep {add_noise_timestep} is too large, max_train_t is {max_train_t}"

        # 2. 设置去噪时间表
        denoising_steps = int(self.hparams.n_timestep * denoising_rate)
        self.scheduler.set_timesteps(denoising_steps, device=self.device)

        # 3. 对输入图像添加精确水平的噪声
        noise = torch.randn_like(input_image)
        timesteps_for_noise = torch.tensor([start_t], device=self.device)
        noisy_image = self.scheduler.add_noise(input_image, noise, timesteps_for_noise)
        
        # 3. 确定去噪循环的起点
        #    在 `denoising_steps` 生成的稀疏时间表中，找到第一个 <= start_t 的位置。
        #    例如，如果 start_t=199，而 scheduler.timesteps 是 [999, 979, ..., 219, 199, 179, ...],
        #    那么循环就从 t=199 这个位置开始。
        try:
            initial_loop_step = next(
                i for i, t in enumerate(self.scheduler.timesteps) if t <= start_t
            )
        except StopIteration:
            # 如果去噪步数太少，可能时间表里没有任何一个 t <= start_t，这不合理
            raise ValueError(f"无法在 {denoising_steps} 步的去噪计划中找到起点 t={start_t}。")
        
        # 4. 获取 EMA 模型用于推理
        model_to_use = self.ema.module
        model_to_use.eval()
        
        # 6. 去噪循环
        image = noisy_image # <-- FIX: 初始化 image 变量
        for i, t in enumerate(self.scheduler.timesteps[initial_loop_step:]):
            timestep_batch = t.unsqueeze(0).repeat(image.shape[0])
            
            # 预测噪声
            model_output = model_to_use(image, timestep_batch)
            
            # 使用 scheduler.step 计算上一步的图像
            # DDPM scheduler 不需要 prev_t，它会根据当前 t 和自身的内部状态计算
            scheduler_output = self.scheduler.step(model_output, t, image)
            image = scheduler_output.prev_sample

        return image, noisy_image

    def on_validation_epoch_end(self):
        final_metrics = self.metrics.compute()
        org_metrics = self.org_metrics.compute()
        show_images = torchvision.utils.make_grid(self.validation_outputs, value_range=(-1, 1),nrow=len(self.validation_outputs))
        show_images = (show_images + 1) / 2
        self.logger.experiment.add_image("VAL/val_images", show_images, self.global_step)
        self.log("VAL/val_ssim", final_metrics["StructuralSimilarityIndexMeasure"], prog_bar=True, logger=True)
        self.log("VAL/val_mse", final_metrics["MeanSquaredError"], prog_bar=True, logger=True)
        self.log("VAL/org_ssim", org_metrics["StructuralSimilarityIndexMeasure"], prog_bar=True, logger=True)
        self.log("VAL/org_mse", org_metrics["MeanSquaredError"], prog_bar=True, logger=True)
        self.validation_outputs = []
        self.metrics.reset()
        self.org_metrics.reset()

    # test
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        bs = batch['pixel_values'].shape[0]
        o0 = batch['pixel_values']
        o1 = batch['condition']
        file_name = batch['file_name']

        # 如果使用亮度调整网络，需要先转换输入
        if self.hparams.brightness_net:
            brightness_params_i = torch.cat([o1.mean(dim=(1,2,3)).unsqueeze(1), o1.std(dim=(1,2,3)).unsqueeze(1)], dim=1)
            x1 = self.brightness_net_f(o1, brightness_params_i)
        else:
            x1 = o1
        
        # 调用修正后的推理函数，并明确指定 "精细" 去噪步数
        pred, _ = self.sdedit_inference(
            x1, 
            add_noise_timestep=self.hparams.edit_timestep, # 加 200 步的噪声
            denoising_rate=1.0 # Test: 使用 1000 步进行精细去噪
        )

        # 如果使用亮度调整网络，需要将结果转换回原始亮度空间
        if self.hparams.brightness_net:
            pred = pred.to(self.device)
            out_mean = torch.tensor([[0.087878]], device=self.device)
            out_std = torch.tensor([[0.309309]], device=self.device)
            brightness_params_o = torch.cat([out_mean, out_std], dim=1)
            brightness_params_o = brightness_params_o.repeat(bs, 1)
            pred = self.brightness_net_b(pred, brightness_params_o)
        
        self.metrics.update(pred.detach().cpu() / 2 + 0.5, o0.detach().cpu() / 2 + 0.5)
        # 保存图片
        save_dir = os.path.join(self.hparams.log_dir, self.hparams.name, "test")
        os.makedirs(save_dir, exist_ok=True)
        for i in range(bs):
            torchvision.utils.save_image(pred[i], f"{save_dir}/{file_name[i]}.png", normalize=True, value_range=(-1, 1))

    def on_test_epoch_end(self):
        final_metrics = self.metrics.compute()
        print(f"SSIM: {final_metrics['StructuralSimilarityIndexMeasure']}, MSE: {final_metrics['MeanSquaredError']}")
        self.metrics.reset()

# checkpoint 
    def on_save_checkpoint(self, checkpoint):
        # 保存 EMA 状态（包含所有网络的 EMA 参数）
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        ema_state_dict = checkpoint.get("ema")
        if ema_state_dict:
            self.ema.load_state_dict(ema_state_dict)