# ===== 3. models/simplified_baselines/model.py =====
"""
GAMUS nDSM预测模型 - 基于DINOv2的简化版本
主模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union

from ..base_model import BaseDepthModel
from ..losses.multi_scale_loss import SingleScaleLoss
from .decoder import SimplifiedDPTHead, SimpleNDSMHead

# 尝试导入DINOv2相关模块
try:
    from .dinov2 import DINOv2
    DINOV2_AVAILABLE = True
    print("import dinov2")
except ImportError:
    DINOV2_AVAILABLE = False
    print("警告: DINOv2模块不可用，将使用占位符实现")

class GAMUSNDSMPredictor(BaseDepthModel):
    """GAMUS nDSM预测模型 - 简化版本"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        model_config = config.get('model_config', {})
        
        # 模型参数
        self.encoder_type = model_config.get('encoder', 'vitb')
        self.features = model_config.get('features', 256)
        self.use_pretrained_dpt = model_config.get('use_pretrained_dpt', True)
        
        # ViT编码器层索引
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
        }
        
        # 创建编码器
        if DINOV2_AVAILABLE and self.encoder_type in ['vits', 'vitb', 'vitl']:
            self.pretrained = DINOv2(model_name=self.encoder_type)
            self.embed_dim = self.pretrained.embed_dim
        else:
            raise ImportError("DINOv2模块不可用或编码器类型不支持")
        
        # 创建解码器组件
        if self.use_pretrained_dpt:
            self.depth_head = SimplifiedDPTHead(self.embed_dim, self.features)
        
        # nDSM预测头
        self.ndsm_head = SimpleNDSMHead(input_channels=1)
        
        # 损失函数
        loss_config = config.get('loss_config', {'base_loss': 'l1'})
        loss_type = loss_config.get('base_loss', 'l1')
        
        if loss_type == 'l1':
            self.loss_fn = SingleScaleLoss('l1')
        elif loss_type == 'mse':
            self.loss_fn = SingleScaleLoss('mse')
        elif loss_type == 'smooth_l1':
            self.loss_fn = SingleScaleLoss('smooth_l1')
        elif loss_type == 'huber':
            self.loss_fn = SingleScaleLoss('huber')

        else:
            print(f"警告: 未知损失类型 {loss_type}，使用默认 smooth_l1")
            self.loss_fn = SingleScaleLoss('smooth_l1')
        
        # 加载预训练权重
        pretrained_path = model_config.get('pretrained_path')
        if pretrained_path and Path(pretrained_path).exists():
            self.load_pretrained_weights(pretrained_path)
        elif pretrained_path:
            print(f"警告: 预训练权重文件不存在: {pretrained_path}")
        
        # 冻结编码器
        if model_config.get('freeze_encoder', True):
            self.freeze_encoder()
    
    def load_pretrained_weights(self, pretrained_path: str):
        """加载预训练权重"""
        try:
            print(f"正在加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', checkpoint)
            else:
                state_dict = checkpoint
            
            # 只加载匹配的权重
            model_dict = self.state_dict()
            pretrained_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
            
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            
            print(f"成功加载预训练权重: {len(pretrained_dict)}/{len(model_dict)} 个参数")
                
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.pretrained.parameters():
            param.requires_grad = False
        
        if self.use_pretrained_dpt:
            for param in self.depth_head.parameters():
                param.requires_grad = False
        
        print("编码器参数已冻结")
    
    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.pretrained.parameters():
            param.requires_grad = True
        
        if self.use_pretrained_dpt:
            for param in self.depth_head.parameters():
                param.requires_grad = True
        
        print("编码器参数已解冻")
    
    def forward(self, 
               x: torch.Tensor, 
               return_multi_scale: bool = False,
               **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            return_multi_scale: 是否返回多尺度结果（此模型忽略此参数）
            **kwargs: 其他参数
        Returns:
            nDSM预测结果 [B, H, W]
        """
        batch_size = x.shape[0]
        
        # 448x448输入，patch大小14x14，所以patch_h=patch_w=32
        patch_h = patch_w = 32
        
        # 特征提取
        features = self.pretrained.get_intermediate_layers(
            x, 
            self.intermediate_layer_idx[self.encoder_type],
            return_class_token=True
        )
        
        if self.use_pretrained_dpt:
            # 使用简化DPT头
            depth = self.depth_head(features, patch_h, patch_w)
            depth = F.relu(depth)  # 确保非负
        else:
            # 简化版本：直接处理最后一层特征
            last_feature = features[-1]
            if isinstance(last_feature, tuple):
                last_feature = last_feature[0]  # 移除class token
            
            if last_feature.dim() == 3:  # (B, N, C)
                last_feature = last_feature.permute(0, 2, 1).reshape(
                    batch_size, -1, patch_h, patch_w
                )
            
            depth = F.interpolate(
                last_feature, 
                size=(448, 448), 
                mode='bilinear', 
                align_corners=False
            )
            # 简单投影到单通道
            if depth.shape[1] > 1:
                depth = F.conv2d(
                    depth, 
                    torch.ones(1, depth.shape[1], 1, 1, device=depth.device) / depth.shape[1],
                    bias=None
                )
            depth = F.relu(depth)
        
        # nDSM预测头
        ndsm_pred = self.ndsm_head(depth)
        
        return ndsm_pred
    
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    targets: torch.Tensor,
                    masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            predictions: 模型预测结果 [B, H, W]
            targets: 真实标签 [B, H, W] 
            masks: 可选掩码 [B, H, W]
        Returns:
            损失字典，包含 'total_loss' 等键
        """
        return self.loss_fn(predictions, targets, masks)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            'encoder_type': self.encoder_type,
            'use_pretrained_dpt': self.use_pretrained_dpt,
            'features': self.features,
            'embed_dim': getattr(self, 'embed_dim', 'unknown'),
        })
        return info

def create_gamus_ndsm_model(config: Dict[str, Any]) -> GAMUSNDSMPredictor:
    """创建GAMUS nDSM模型的工厂函数"""
    return GAMUSNDSMPredictor(config)
