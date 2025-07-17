# ===== 2. models/simplified_baselines/decoder.py =====
"""
GAMUS简化模型解码器组件
包含SimplifiedDPTHead和SimpleNDSMHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

class SimpleNDSMHead(nn.Module):
    """简单的nDSM预测头"""
    def __init__(self, input_channels=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=False)  # 确保非负输出
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            nDSM预测结果 [B, H, W]
        """
        return self.layers(x).squeeze(1)  # 移除通道维度

class SimplifiedDPTHead(nn.Module):
    """简化的DPT头部"""
    def __init__(self, in_channels: int, features: int = 256):
        super().__init__()
        
        self.in_channels = in_channels
        self.features = features
        
        # 简化的特征处理
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU(inplace=True),
        )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(features//2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=False)
        )

    def forward(self, 
               features: Union[List, torch.Tensor], 
               patch_h: int, 
               patch_w: int) -> torch.Tensor:
        """
        处理特征并输出深度图
        
        Args:
            features: DINOv2特征列表或单个特征张量
            patch_h: patch高度
            patch_w: patch宽度
        Returns:
            深度预测图 [B, 1, H, W]
        """
        # 取最后一层特征
        if isinstance(features, list):
            x = features[-1]
            # DINOv2返回的是(patch_features, class_token)元组
            if isinstance(x, tuple):
                x = x[0]  # 取patch features，忽略class token
        else:
            x = features
        
        # 如果是token格式，转换为conv格式
        if x.dim() == 3:  # [B, N, C]
            B, N, C = x.shape
            x = x.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
        
        # 特征处理
        x = self.feature_conv(x)
        
        # 上采样到目标尺寸（448x448）
        x = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=True)
        
        # 输出预测
        out = self.output_conv(x)
        return out