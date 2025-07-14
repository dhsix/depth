
from .im2height_model import Im2HeightModel

def create_im2height_model(config):
    """创建Im2Height模型的工厂函数"""
    return Im2HeightModel(config)

__all__ = ['Im2HeightModel', 'create_im2height_model']