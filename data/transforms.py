import cv2
import numpy as np
import torch
import torchvision.transforms as T
from typing import Dict, Any, Optional, Tuple

class Resize:
    """保持长宽比的resize，确保是14的倍数"""
    def __init__(self, 
                 width: int, 
                 height: int, 
                 resize_target: bool = False,
                 keep_aspect_ratio: bool = True, 
                 ensure_multiple_of: int = 14,
                 resize_method: str = 'lower_bound',
                 image_interpolation_method: int = cv2.INTER_CUBIC):
        self.width = width
        self.height = height
        self.resize_target = resize_target
        self.keep_aspect_ratio = keep_aspect_ratio
        self.multiple_of = ensure_multiple_of
        self.resize_method = resize_method
        self.image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x: int, min_val: int = 0, max_val: Optional[int] = None) -> int:
        """确保x是multiple_of的倍数"""
        y = (np.round(x / self.multiple_of) * self.multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.multiple_of) * self.multiple_of).astype(int)
        if y < min_val:
            y = (np.ceil(x / self.multiple_of) * self.multiple_of).astype(int)
        return y

    def get_size(self, width: int, height: int) -> Tuple[int, int]:
        """计算输出尺寸"""
        scale_height = self.height / height
        scale_width = self.width / width

        if self.keep_aspect_ratio:
            if self.resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height

        new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.height)
        new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.width)

        return (new_width, new_height)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        
        sample["image"] = cv2.resize(
            sample["image"], 
            (width, height), 
            interpolation=self.image_interpolation_method
        )
        
        if self.resize_target and "depth" in sample:
            sample["depth"] = cv2.resize(
                sample["depth"], 
                (width, height), 
                interpolation=cv2.INTER_NEAREST
            )
        # 新增：处理mask数据
        if "mask" in sample:
            sample["mask"] = cv2.resize(
                sample["mask"], 
                (width, height), 
                interpolation=cv2.INTER_NEAREST  # mask用最近邻插值
            )
        
        # 新增：处理语义mask数据
        if "semantic_mask" in sample:
            sample["semantic_mask"] = cv2.resize(
                sample["semantic_mask"], 
                (width, height), 
                interpolation=cv2.INTER_NEAREST  # 语义标签必须用最近邻插值
            )
        
        return sample

class NormalizeImage:
    """图像标准化"""
    def __init__(self, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample["image"] = (sample["image"] - self.mean) / self.std
        return sample

class PrepareForNet:
    """准备输入网络的格式"""
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)
        
        if "depth" in sample:
            sample["depth"] = np.ascontiguousarray(sample["depth"]).astype(np.float32)
        if "mask" in sample:
            sample["mask"] = np.ascontiguousarray(sample["mask"]).astype(np.float32)
        
        if "semantic_mask" in sample:
            sample["semantic_mask"] = np.ascontiguousarray(sample["semantic_mask"]).astype(np.uint8)
        
        return sample

class ToTensor:
    """转换为Tensor"""
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for key in sample:
            if key in ["image", "depth", "mask","semantic_mask"]:
                sample[key] = torch.from_numpy(sample[key])
        return sample

class RandomHorizontalFlip:
    """随机水平翻转"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if np.random.random() < self.prob:
            sample["image"] = np.fliplr(sample["image"]).copy()
            if "depth" in sample:
                sample["depth"] = np.fliplr(sample["depth"]).copy()
                        # 🔥 需要添加：处理mask
            if "mask" in sample:
                sample["mask"] = np.fliplr(sample["mask"]).copy()
            # 🔥 需要添加：处理语义mask
            if "semantic_mask" in sample:
                sample["semantic_mask"] = np.fliplr(sample["semantic_mask"]).copy()
        return sample

class ColorJitter:
    """颜色抖动"""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2, hue: float = 0.1):
        self.transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 转换为PIL格式进行ColorJitter
        image = sample["image"]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        from PIL import Image
        pil_image = Image.fromarray(image)
        jittered = self.transform(pil_image)
        sample["image"] = np.array(jittered).astype(np.float32) / 255.0
        
        return sample

def get_transforms(config: Dict[str, Any], is_training: bool = True) -> T.Compose:
    """获取数据变换pipeline"""
    transforms = []
    
    if is_training:
        # 训练时的数据增强
        if config.get('horizontal_flip', 0) > 0:
            transforms.append(RandomHorizontalFlip(config['horizontal_flip']))
        
        if config.get('color_jitter', 0) > 0:
            transforms.append(ColorJitter())
    
    # 基础变换
    transforms.extend([
        Resize(width=config.get('input_size', 448), 
               height=config.get('input_size', 448),
               resize_target=True,
               keep_aspect_ratio=True,
               ensure_multiple_of=config.get('patch_size', 14)),
        NormalizeImage(),
        PrepareForNet(),
        ToTensor()
    ])
    
    return T.Compose(transforms)