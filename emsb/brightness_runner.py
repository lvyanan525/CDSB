import os
import torch
import numpy as np
import torchvision
import torchmetrics
import lightning as L
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
        if opt.brightness_net and opt.brightness_type != 'naive':
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
            x1 = bright_norm(o1, net_f=self.brightness_net_f, type=self.hparams.brightness_type)
        else:
            x1 = o1
        
        xs, _ = self.ddpm_sampling(x1, self.ema.module, nfe=100, verbose=False)

        pred = xs[:, 0]
        # 如果使用亮度调整网络，需要将结果转换回原始亮度空间
        if self.hparams.brightness_net:
            pred = pred.to(self.device)
            pred = bright_recon(pred, net_b=self.brightness_net_b, type=self.hparams.brightness_type)

        xs = xs.reshape(bs, -1, *o0.shape[1:]) # [bs, n_timestep//100, *o0.shape[1:]]
        if batch_idx == 0:
            x_show = [x.detach().cpu() for x in xs[0]]
            x_show = x_show[::-1]
            if self.hparams.brightness_net:
                self.validation_outputs = [o1[0].detach().cpu(), *x_show, pred[0].detach().cpu(), o0[0].detach().cpu()]
            else:
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
        # import torchmetrics.functional as F
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
        
        # for i in range(bs):
        #     ssim_i = F.structural_similarity_index_measure(pred[i].unsqueeze(0)/2+0.5, o0[i].unsqueeze(0)/2+0.5, data_range=1.0)
        #     print(f"{file_name[i]} ssim: {ssim_i}")
        self.metrics.update(pred.detach().cpu() / 2 + 0.5, o0.detach().cpu() / 2 + 0.5)
        # pred = pred * 0.5 + 0.5
        # pred = pred.clamp(0, 1)
        # 保存图片
        save_dir = os.path.join(self.hparams.log_dir, self.hparams.name, "test")
        os.makedirs(save_dir, exist_ok=True)
        for i in range(bs):
            torchvision.utils.save_image(pred[i], f"{save_dir}/{file_name[i]}.png", normalize=True, value_range=(-1, 1))

    def on_test_epoch_end(self):
        final_metrics = self.metrics.compute()
        print(final_metrics)
        self.metrics.reset()


# checkpoint 
    def on_save_checkpoint(self, checkpoint):
        # 保存 EMA 状态（包含所有网络的 EMA 参数）
        checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        ema_state_dict = checkpoint.get("ema")
        if ema_state_dict:
            self.ema.load_state_dict(ema_state_dict)