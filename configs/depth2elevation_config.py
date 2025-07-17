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
        # 'freeze_encoder': False,  # 是否冻结编码器
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
    # 冻结策略配置
    freezing_config: Dict[str, Any] = field(default_factory=lambda: {
        'strategy': 'selective',  # 'none', 'simple', 'selective'
        'freeze_patch_embed': True,  # 是否冻结patch embedding
        'unfreeze_positional': True,  # 是否解冻位置编码
        'print_stats': True,  # 是否打印参数统计信息
    })

    # 数据增强
    augmentation_config: Dict[str, Any] = field(default_factory=lambda: {
        'horizontal_flip': 0.5,
        'color_jitter': 0.5,
        'gaussian_blur': 0.5,
    })
    # 新增：分布重加权配置
    reweight_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable': False,
        'num_height_bins': 50,
        'max_height': 100.0,
        'alpha': 0.7,
        'lds_kernel': 'gaussian',
        'lds_ks': 5,
        'lds_sigma': 2.0,
        'base_loss': 'smooth_l1',
        'scale_weights': {
            'scale_1': 0.125,
            'scale_2': 0.25, 
            'scale_3': 0.5,
            'scale_4': 1.0
        },
        'focal_params': {
            'alpha': 0.25,
            'gamma': 2.0,
            'height_threshold': 20.0
        }
    })

