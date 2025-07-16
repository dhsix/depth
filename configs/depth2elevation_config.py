from dataclasses import dataclass, field
from typing import Dict, Any
from .base_config import BaseConfig

@dataclass
class Depth2ElevationConfig(BaseConfig):
    """Depth2Elevation模型配置"""
    
    model_name: str = "depth2elevation"
    
    # 模型特定配置
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        'encoder': 'vitb',  # vits, vitb, vitl, vitg
        'img_size': 448,
        'patch_size': 14,
        'pretrained_path': None,  # DAM预训练权重路径
        'freeze_encoder': False,  # 是否冻结编码器
    })
    
    # 损失函数配置
    loss_config: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'multi_scale_loss',
        'gamma': 1.0,     # L_ai权重
        'delta': 1.0,     # L_si权重  
        'mu': 0.05,       # L_grad权重
        'beta': 0.15,     # SI loss中的beta
        'epsilon': 1e-7,  # log计算中的epsilon
        'lambda_grad': 1e-3,  # 梯度损失权重
    })
    
    # 训练特定设置
    learning_rate: float = 5e-6
    num_epochs: int = 50
    use_multi_scale_output: bool = True
    
    compute_metrics_interval:int = 1
    log_scale_losses: bool=True

    # 数据增强
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
        'horizontal_flip': 0.5,
        'color_jitter': 0.5,
        'gaussian_blur': 0.5,
    })