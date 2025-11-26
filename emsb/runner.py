import itertools
import os
import torch
import numpy as np
import torchvision
import torchmetrics
import lightning as L
import json
from emsb.network import Image256Net
from emsb.FiLMLayer import BrightNormNet
from timm.utils import ModelEmaV2
from emsb.SchrodingerBridge import Diffusion, make_beta_schedule
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR
from emsb.blurloss import blur_loss
from emsb.brightnessloss import bright_loss, bright_norm, bright_recon
import torch.nn.functional as F

class emsb_runner(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters(opt)

        # 构建diffusion
        betas = make_beta_schedule(n_timestep=self.hparams.n_timestep, linear_end=self.hparams.beta_max / self.hparams.n_timestep)
        betas = np.concatenate([betas[:self.hparams.n_timestep//2], np.flip(betas[:self.hparams.n_timestep//2])])
        self.diffusion = Diffusion(betas)
        noise_levels = torch.linspace(self.hparams.t0, self.hparams.T, self.hparams.n_timestep) * self.hparams.n_timestep
        # 构建网络和ema
        self.net = Image256Net(noise_levels=noise_levels, use_fp16=self.hparams.use_fp16, cond=self.hparams.cond_x1)
        
        # 初始化亮度调整网络（如果需要）
        if self.hparams.brightness_net:
            if self.hparams.brightness_type == 'vae':
                from diffusers import AutoencoderKL
                self.brightness_net_vae = AutoencoderKL.from_pretrained('ckpts', subfolder='vae')
                self.brightness_net_f = self.brightness_net_vae.encode
                self.brightness_net_b = self.brightness_net_vae.decode
                
                # 修改网络输入和输出层以支持4通道VAE
                self._modify_conv_layers_for_vae()
            else:
                self.brightness_net_f = BrightNormNet(brightness_param_dim=2, input_channels=getattr(self.hparams, 'brightness_channel', 1))
                self.brightness_net_b = BrightNormNet(brightness_param_dim=2, input_channels=getattr(self.hparams, 'brightness_channel', 1))

        if getattr(self.hparams, 'blur_loss', 0.0) > 0.0:
            self.blur_loss = blur_loss()

        self.ema = ModelEmaV2(self.net, decay=self.hparams.ema)

        # 构建metrics
        self.validation_outputs = []
        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0),
            # MSE
            torchmetrics.MeanSquaredError()
        ])
    
    def _modify_conv_layers_for_vae(self):
        """修改已创建网络的第一层和最后一层conv以支持4通道VAE输入输出"""
        
        # 1. 修改第一层conv（输入层）
        first_conv = self.net.diffusion_model.input_blocks[0][0]
        
        if first_conv.in_channels == 3:
            # 获取原始权重
            old_weight = first_conv.weight.data.clone()  # [out_ch, 3, 3, 3]
            
            # 创建新的4通道conv层
            new_conv = torch.nn.Conv2d(
                in_channels=4,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # 初始化新权重
            with torch.no_grad():
                # 复制原有的3通道权重
                new_conv.weight.data[:, :3, :, :] = old_weight
                
                # 第4通道使用RGB均值初始化
                new_conv.weight.data[:, 3:4, :, :] = old_weight.mean(dim=1, keepdim=True)
                
                # 复制bias（如果存在）
                if first_conv.bias is not None:
                    new_conv.bias.data = first_conv.bias.data.clone()
            
            # 替换第一层conv
            self.net.diffusion_model.input_blocks[0][0] = new_conv
            
            print(f"已将第一层conv从{old_weight.shape}扩展到{new_conv.weight.shape}")
        else:
            print(f"第一层conv已经是{first_conv.in_channels}通道，无需修改")
        
        # 2. 修改最后一层conv（输出层）
        # UNet的输出层在self.out中的最后一个conv
        output_conv = self.net.diffusion_model.out[-1]  # 最后一个是conv层
        
        if output_conv.out_channels == 3:
            # 获取原始权重
            old_out_weight = output_conv.weight.data.clone()  # [3, in_ch, 3, 3]
            
            # 创建新的4通道输出conv层
            new_out_conv = torch.nn.Conv2d(
                in_channels=output_conv.in_channels,
                out_channels=4,
                kernel_size=output_conv.kernel_size,
                stride=output_conv.stride,
                padding=output_conv.padding,
                bias=output_conv.bias is not None
            )
            
            # 初始化新权重
            with torch.no_grad():
                # 复制原有的3通道权重
                new_out_conv.weight.data[:3, :, :, :] = old_out_weight
                
                # 第4通道使用RGB均值初始化
                new_out_conv.weight.data[3:4, :, :, :] = old_out_weight.mean(dim=0, keepdim=True)
                
                # 复制bias（如果存在）
                if output_conv.bias is not None:
                    new_bias = torch.zeros(4)
                    new_bias[:3] = output_conv.bias.data.clone()
                    new_bias[3] = output_conv.bias.data.mean()  # 第4通道bias用均值初始化
                    new_out_conv.bias.data = new_bias
            
            # 替换最后一层conv
            self.net.diffusion_model.out[-1] = new_out_conv
            
            print(f"已将输出层conv从{old_out_weight.shape}扩展到{new_out_conv.weight.shape}")
        else:
            print(f"输出层conv已经是{output_conv.out_channels}通道，无需修改")
            
    def training_step(self, batch, batch_idx):
        opt = self.hparams
        o0 = batch['pixel_values']
        o1 = batch['condition']

        if opt.brightness_net:
            x0, x1, loss_recon, loss_norm, loss_bright_var, loss_brightness = bright_loss(self.brightness_net_f, self.brightness_net_b, o0, o1, opt.brightness_type)
        else:
            x0 = o0
            x1 = o1
            loss_brightness = 0

        timesteps = torch.randint(0, self.hparams.n_timestep, (x0.shape[0],))
        xt = self.diffusion.q_sample(timesteps, x0, x1, ot_ode=opt.ot_ode)
        target = self.diffusion.compute_label(timesteps, x0, xt)

        pred = self.net(xt, timesteps)
        if opt.blur_loss:
            loss_blur = self.blur_loss(x1, pred)
        else:
            loss_blur = 0

       
        loss_base = F.mse_loss(pred, target)
        loss = loss_base + loss_brightness + self.hparams.blur_loss * loss_blur

        self.log('TRAIN/loss_base', loss_base, prog_bar=True)
        self.log('TRAIN/train_loss', loss, prog_bar=True)
        if opt.brightness_net and opt.brightness_type in ['l2', 'stl']:
            self.log('TRAIN/loss_norm', loss_norm, prog_bar=True)
            self.log('TRAIN/loss_recon', loss_recon, prog_bar=True)
            self.log('TRAIN/loss_bright_var', loss_bright_var, prog_bar=True)
            self.log('TRAIN/brightness_loss', loss_brightness, prog_bar=True)
        if opt.blur_loss:
            self.log('TRAIN/loss_blur', loss_blur, prog_bar=True)
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        self.ema.update(self.net)
       
    # def on_train_epoch_end(self):
    #     print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB in epoch {self.current_epoch}")
    #     print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB in epoch {self.current_epoch}")
    
    def configure_optimizers(self):
        """专业推荐配置：AdamW + 线性Warmup + 余弦退火"""
        opt = self.hparams
        
        # 收集所有需要优化的参数
        param_groups = [
            {'params': self.net.parameters(), 'lr': opt.lr}
        ]
        if opt.brightness_net and opt.brightness_type != 'vae':
            brightness_params = itertools.chain(self.brightness_net_f.parameters(), self.brightness_net_b.parameters())
            param_groups.append({'params': brightness_params, 'lr': opt.lr_brightness})
            
        
        optimizer = torch.optim.AdamW(
            param_groups
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
            x1 = bright_norm(o1, net_f=self.brightness_net_f, type=self.hparams.brightness_type)
        else:
            x1 = o1
        
        xs, _ = self.ddpm_sampling(x1, self.ema.module, nfe=20, verbose=False)

        pred = xs[:, 0]
        # 如果使用亮度调整网络，需要将结果转换回原始亮度空间
        if self.hparams.brightness_net:
            pred = pred.to(self.device)
            pred = bright_recon(pred, net_b=self.brightness_net_b, type=self.hparams.brightness_type)

        xs = xs.reshape(bs, -1, *x1.shape[1:]) # [bs, n_timestep//100, *o0.shape[1:]]
        if batch_idx == 0:
            if self.hparams.brightness_net:
                if self.hparams.brightness_type == 'vae':
                    # VAE模式下，中间结果是4通道，只显示前3个通道用于可视化
                    xs_ = xs[0,:,:3] # [n_timestep//100, 3, H, W]
                    x_show = [F.interpolate(x.unsqueeze(0), size=o0.shape[-2:], mode='bilinear', align_corners=False).squeeze(0).detach().cpu() for x in xs_]
                    x_show = x_show[::-1]  # 反转顺序以正确显示diffusion过程
                    self.validation_outputs = [o1[0].detach().cpu(), *x_show, pred[0].detach().cpu(), o0[0].detach().cpu()]
                else:
                    x_show = [x.detach().cpu() for x in xs[0]]
                    x_show = x_show[::-1]
                    self.validation_outputs = [o1[0].detach().cpu(), *x_show, pred[0].detach().cpu(), o0[0].detach().cpu()]
            else:
                x_show = [x.detach().cpu() for x in xs[0]]
                x_show = x_show[::-1]
                self.validation_outputs = [o1[0].detach().cpu(), *x_show, o0[0].detach().cpu()]

        self.metrics.update(pred.detach().cpu() / 2 + 0.5, o0.detach().cpu() / 2 + 0.5)

    @torch.no_grad()
    def ddpm_sampling(self, x1, net, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):
        opt = self.hparams
        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        def space_indices(num_steps, count):
            assert count <= num_steps

            if count <= 1:
                frac_stride = 1
            else:
                frac_stride = (num_steps - 1) / (count - 1)

            cur_idx = 0.0
            taken_steps = []
            for _ in range(count):
                taken_steps.append(round(cur_idx))
                cur_idx += frac_stride
            return taken_steps
        
        nfe = nfe or opt.n_timestep-1
        assert 0 < nfe < opt.n_timestep == len(self.diffusion.betas)
        steps = space_indices(opt.n_timestep, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0    
        if verbose:
            print("DDPM Sampling: steps={}, nfe={}, log_steps={}".format(opt.n_timestep, nfe, log_steps)  )

        x1 = x1.to(self.device)
        if cond is not None: cond = cond.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        net.eval()
        def pred_x0_fn(xt, step):
            step = torch.full((xt.shape[0],), step, dtype=torch.long)
            out = net(xt, step, cond=cond)
            return self.diffusion.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

        xs, pred_x0 = self.diffusion.ddpm_sampling(
            steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
        )
        net.train()

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    def on_validation_epoch_end(self):
        final_metrics = self.metrics.compute()
        show_images = torchvision.utils.make_grid(self.validation_outputs, value_range=(-1, 1),nrow=len(self.validation_outputs))
        show_images = (show_images + 1) / 2
        self.logger.experiment.add_image("VAL/val_images", show_images, self.global_step)
        self.log("VAL/val_ssim", final_metrics["StructuralSimilarityIndexMeasure"], prog_bar=True, logger=True)
        self.log("VAL/val_mse", final_metrics["MeanSquaredError"], prog_bar=True, logger=True)

        self.validation_outputs = []
        self.metrics.reset()

# test
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        bs = batch['pixel_values'].shape[0]
        o0 = batch['pixel_values']
        o1 = batch['condition']
        file_name = batch['file_name']
        # 如果使用亮度调整网络，需要先转换输入
        if self.hparams.brightness_net:
            x1 = bright_norm(o1, net_f=self.brightness_net_f, type=self.hparams.brightness_type)
        else:
            x1 = o1
        
        xs, _ = self.ddpm_sampling(x1, self.ema.module, nfe=self.hparams.nfe, verbose=False)

        pred = xs[:, 0]
        # 如果使用亮度调整网络，需要将结果转换回原始亮度空间
        if self.hparams.brightness_net:
            pred = pred.to(self.device)
            pred = bright_recon(pred, net_b=self.brightness_net_b, type=self.hparams.brightness_type)
        
        xs = xs.reshape(bs, -1, *x1.shape[1:]) # [bs, n_timestep//100, *x1.shape[1:]]

        self.metrics.update(pred.detach().cpu() / 2 + 0.5, o0.detach().cpu() / 2 + 0.5)

        # 保存图片
        eval_root = getattr(self.hparams, 'eval_save_dir', None)
        if eval_root is None:
            eval_root = os.path.join(self.hparams.log_dir, self.hparams.name, "test")
        images_dir = os.path.join(eval_root, "images")
        os.makedirs(images_dir, exist_ok=True)
        for i in range(bs):
            if not file_name[i].endswith(".bmp"):
                file_name[i] = file_name[i] + ".bmp"
            torchvision.utils.save_image(pred[i], f"{images_dir}/{file_name[i]}", normalize=True, value_range=(-1, 1))

    def on_test_epoch_end(self):
        final_metrics = self.metrics.compute()
        # 打印并保存聚合指标
        print(final_metrics)

        # 将指标保存到统一目录（如提供）
        eval_root = getattr(self.hparams, 'eval_save_dir', None)
        if eval_root is not None:
            os.makedirs(eval_root, exist_ok=True)
            metrics_to_save = {
                'val_ssim': float(final_metrics["StructuralSimilarityIndexMeasure"].item()),
                'val_mse': float(final_metrics["MeanSquaredError"].item()),
            }
            # 附加nfe信息（如有）
            if hasattr(self.hparams, 'nfe') and self.hparams.nfe is not None:
                metrics_to_save['nfe'] = int(self.hparams.nfe)
            metrics_path = os.path.join(eval_root, 'metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)
        self.metrics.reset()

    @torch.no_grad()
    def inference(self, o1, nfe=None):
        # 如果使用亮度调整网络，需要先转换输入
        if self.hparams.brightness_net:
            x1 = bright_norm(o1, net_f=self.brightness_net_f, type=self.hparams.brightness_type)
        else:
            x1 = o1
        
        xs, _ = self.ddpm_sampling(x1, self.ema.module, nfe=nfe, verbose=False)

        pred = xs[:, 0]
        # 如果使用亮度调整网络，需要将结果转换回原始亮度空间
        if self.hparams.brightness_net:
            pred = pred.to(self.device)
            pred = bright_recon(pred, net_b=self.brightness_net_b, type=self.hparams.brightness_type)

        return pred
    
    # # checkpoint 
    # def on_save_checkpoint(self, checkpoint):
    #     # 保存 EMA 状态（包含所有网络的 EMA 参数）
    #     checkpoint["ema"] = self.ema.state_dict()
    #     checkpoint["brightness_net_f"] = self.brightness_net_f.state_dict()
    #     checkpoint["brightness_net_b"] = self.brightness_net_b.state_dict()

    # def on_load_checkpoint(self, checkpoint):
    #     ema_state_dict = checkpoint.get("ema")
    #     if ema_state_dict:
    #         self.ema.load_state_dict(ema_state_dict)