from typing import Dict,Any
from .base_model import BaseDepthModel
from .depth2elevation import create_depth2elevation_model
from .baselines.depth_anything import create_depth_anything_model
from .GradientAdaptiveEdgeModule import create_depth2elevation_gra_model
from .imele import create_imele_model
from .im2height import create_im2height_model
from .losses import get_loss_function
from .GradientAdaptiveEdgeModule import create_depth2elevation_multiscale_model

def create_model(config: Dict[str, Any]) -> BaseDepthModel:
    """模型工厂函数"""
    model_name = config.get('model_name', 'depth2elevation')
    
    model_creators = {
        'depth2elevation': create_depth2elevation_model,
        'depth_anything_v2': create_depth_anything_model,
        'depth2elevation_gra': create_depth2elevation_gra_model,
        # 'htc_dc': create_htc_dc_model,
        'imele': create_imele_model,
        'im2height': create_im2height_model,
        'depth2elevation_multiscale': create_depth2elevation_multiscale_model,
    }
    
    if model_name not in model_creators:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_creators.keys())}")
    
    return model_creators[model_name](config)

__all__ = [
    'BaseDepthModel',
    'create_model',
    'get_loss_function',
    'create_depth2elevation_model',
    'create_depth2elevation_gra_model',
    'create_depth2elevation_multiscale_model',
    'create_imele_model',
    'create_im2height_model',

]