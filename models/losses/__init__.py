# losses/__init__.py
from .multi_scale_loss import get_loss_function, MultiScaleLoss, SingleScaleLoss
from .base_losses import *  # 如果base_losses里有需要导出的内容

__all__ = [
    'get_loss_function',
    'MultiScaleLoss', 
    'SingleScaleLoss',
    # ... 其他需要导出的内容
]