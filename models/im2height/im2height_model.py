import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union
from ..base_model import BaseDepthModel

class Pool(nn.Module):
    def __init__(self, kernel_size=2, stride=2, **kwargs):
        super(Pool, self).__init__()
        self.pool_fn = nn.MaxPool2d(kernel_size, stride, return_indices=True, **kwargs)

    def forward(self, x, *args, **kwargs):
        x, indices = self.pool_fn(x)
        return x, indices

class Unpool(nn.Module):
    def __init__(self, kernel_size=2, stride=2, **kwargs):
        super(Unpool, self).__init__()
        self.unpool_fn = nn.MaxUnpool2d(kernel_size, stride, **kwargs)

    def forward(self, x, indices, output_size, *args, **kwargs):
        return self.unpool_fn(x, indices=indices, output_size=output_size, *args, **kwargs)

class Block(nn.Module):
    """A Block performs three rounds of conv, batchnorm, relu"""
    def __init__(self, fn, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()

        self.conv1 = fn(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_rest = fn(out_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Identity connection
        self.identity = nn.Sequential()
        if in_channels != out_channels:
            self.identity.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False))
            self.identity.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn(self.conv1(x)))
        y = F.relu(self.bn(self.conv_rest(y)))
        y = self.bn(self.conv_rest(y))
        identity = self.identity(x)
        y = F.relu(y + identity)
        return y

class Im2HeightCore(nn.Module):
    """Im2Height核心网络架构"""
    
    def __init__(self, input_channels=3):
        super(Im2HeightCore, self).__init__()
        
        # 输入通道适配层（原始为单通道，适配RGB）
        if input_channels == 3:
            self.input_conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        else:
            self.input_conv = nn.Identity()
        
        # Convolutions
        self.conv1 = Block(nn.Conv2d, 1, 64)
        self.conv2 = Block(nn.Conv2d, 64, 128)
        self.conv3 = Block(nn.Conv2d, 128, 256)
        self.conv4 = Block(nn.Conv2d, 256, 512)

        # Deconvolutions
        self.deconv1 = Block(nn.ConvTranspose2d, 512, 256)
        self.deconv2 = Block(nn.ConvTranspose2d, 256, 128)
        self.deconv3 = Block(nn.ConvTranspose2d, 128, 64)
        self.deconv4 = Block(nn.ConvTranspose2d, 128, 1)  # residual merge

        self.pool = Pool(2, 2)
        self.unpool = Unpool(2, 2)

    def forward(self, x):
        # 输入通道适配
        x = self.input_conv(x)
        
        # Convolve
        x = self.conv1(x)
        # Residual skip connection
        x_conv_input = x.clone()
        
        x, indices1 = self.pool(x)
        x = self.conv2(x)
        x, indices2 = self.pool(x)
        x = self.conv3(x)
        x, indices3 = self.pool(x)
        x = self.conv4(x)
        x, indices4 = self.pool(x)

        # Deconvolve
        x = self.unpool(x, indices4, x.size())
        x = self.deconv1(x)
        x = self.unpool(x, indices3, x.size())
        x = self.deconv2(x)
        x = self.unpool(x, indices2, x.size())
        x = self.deconv3(x)
        x = self.unpool(x, indices1, x_conv_input.size())

        # Concatenate with residual skip connection
        x = torch.cat((x, x_conv_input), dim=1)
        x = self.deconv4(x)

        return x

class Im2HeightModel(BaseDepthModel):
    """Im2Height模型适配BaseDepthModel"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 配置参数
        self.input_channels = config.get('input_channels', 3)
        self.loss_type = config.get('loss_type', 'combined')
        
        # 构建核心模型
        self.core_model = Im2HeightCore(self.input_channels)
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        return self.core_model(x)
    
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor,
                    masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算损失函数"""
        
        losses = {}
        
        # 处理mask
        if masks is not None:
            valid_mask = masks.bool()
            pred_masked = predictions[valid_mask]
            target_masked = targets[valid_mask]
        else:
            pred_masked = predictions
            target_masked = targets
        
        # 计算损失
        if self.loss_type == 'l1':
            losses['l1_loss'] = F.l1_loss(pred_masked, target_masked)
            losses['total_loss'] = losses['l1_loss']
            
        elif self.loss_type == 'mse':
            losses['mse_loss'] = F.mse_loss(pred_masked, target_masked)
            losses['total_loss'] = losses['mse_loss']
            
        elif self.loss_type == 'combined':
            # Im2Height原始的组合损失
            l1_loss = F.l1_loss(pred_masked, target_masked)
            l2_loss = F.mse_loss(pred_masked, target_masked)
            losses['l1_loss'] = l1_loss
            losses['l2_loss'] = l2_loss
            losses['total_loss'] = l1_loss + 0.1 * l2_loss  # 可调权重
            
        elif self.loss_type == 'ssim_combined':
            # 包含SSIM的损失
            l1_loss = F.l1_loss(pred_masked, target_masked)
            l2_loss = F.mse_loss(pred_masked, target_masked)
            
            # SSIM损失（需要确保张量形状正确）
            if len(predictions.shape) == 3:
                pred_ssim = predictions.unsqueeze(1)
                target_ssim = targets.unsqueeze(1)
            else:
                pred_ssim = predictions
                target_ssim = targets
            
            try:
                from .ssim import ssim
                ssim_loss = 1 - ssim(pred_ssim, target_ssim)
                losses['ssim_loss'] = ssim_loss
                losses['l1_loss'] = l1_loss
                losses['l2_loss'] = l2_loss
                losses['total_loss'] = l1_loss + 0.1 * l2_loss + 0.1 * ssim_loss
            except:
                # 如果SSIM计算失败，回退到组合损失
                losses['l1_loss'] = l1_loss
                losses['l2_loss'] = l2_loss
                losses['total_loss'] = l1_loss + 0.1 * l2_loss
                
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
            
        return losses
    
    def freeze_encoder(self):
        """冻结编码器参数"""
        # 冻结conv层
        for param in self.core_model.conv1.parameters():
            param.requires_grad = False
        for param in self.core_model.conv2.parameters():
            param.requires_grad = False
        for param in self.core_model.conv3.parameters():
            param.requires_grad = False
        for param in self.core_model.conv4.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.core_model.parameters():
            param.requires_grad = True