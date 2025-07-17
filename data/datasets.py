import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Callable, Tuple
import glob
from pathlib import Path

class BaseDepthDataset(Dataset):
    """基础深度估计数据集"""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 transform: Optional[Callable] = None,
                 load_mask: bool = False):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.load_mask = load_mask
        
        self.samples = self._load_samples()
        
    def _load_samples(self) -> list:
        """加载样本列表，子类需要实现"""
        raise NotImplementedError
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0
    
    def _load_depth(self, depth_path: str) -> np.ndarray:
        """加载深度图，子类可以重写"""
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot load depth: {depth_path}")
            # 假设你的 depth 是 16 位，单位是 cm 或 dm，转成 m 范围是合理的
        # 如果是 8 位，需要特别注意
        if depth.dtype == np.uint16:
            depth = depth / 1000.0  # 转成以米为单位，假如你的单位是 mm
        elif depth.dtype == np.uint8:
            depth = depth / 255.0  # 归一化
        return depth
        # return depth.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_info = self.samples[idx]
        
        # 加载图像和深度
        image = self._load_image(sample_info['image_path'])
        depth = self._load_depth(sample_info['depth_path'])
        
        sample = {
            'image': image,
            'depth': depth,
            'image_path': sample_info['image_path'],
            'depth_path': sample_info['depth_path']
        }
        
        # 可选加载mask
        if self.load_mask and 'mask_path' in sample_info:
            mask = cv2.imread(sample_info['mask_path'], cv2.IMREAD_GRAYSCALE)
            sample['mask'] = mask.astype(np.float32) / 255.0
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class GAMUSDataset(BaseDepthDataset):
    """GAMUS数据集"""
    
    def _load_samples(self) -> list:
        # 使用完整的数据集路径
        dataset_root = self.data_root  # 这里data_root应该是完整路径，如 "./datasets/GAMUS"
        split_dir = Path(dataset_root) / self.split
        
        # 数据结构：
        # datasets/GAMUS/
        #   train/
        #     images/
        #     depths/
        #   val/
        #   test/
        
        image_dir = split_dir / "images"
        depth_dir = split_dir / "depths"
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
        
        samples = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            # 对应的深度文件
            # depth_path = depth_dir / f"{image_path.stem}.tif"
            depth_filename=image_path.name.replace("RGB","AGL").replace(".jpg",".png")
            depth_path=depth_dir/depth_filename
            
            if depth_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'depth_path': str(depth_path)
                })
        
        print(f"Loaded {len(samples)} samples from {split_dir}")
        return samples
    def _load_depth(self, depth_path: str) -> np.ndarray:
        """加载GAMUS深度图 - 特殊处理cm单位"""
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Cannot load depth: {depth_path}")
        
        # GAMUS数据集特殊处理：深度单位是cm，需要转换为米
        if depth.dtype == np.uint16:
            # 假设16位深度图，单位是cm，转换为米
            depth_meters = depth.astype(np.float32) / 100.0
        elif depth.dtype == np.uint8:
            # 如果是8位图像，可能需要特殊处理
            max_height_cm = 25500  # 假设最大高度255米(25500cm)
            depth_meters = depth.astype(np.float32) * (max_height_cm / 255.0) / 100.0
        else:
            # 其他数据类型，假设已经是正确的单位
            depth_meters = depth.astype(np.float32)
        
        return depth_meters

class DFC2019Dataset(BaseDepthDataset):
    """DFC2019数据集"""
    
    def _load_samples(self) -> list:
        # DFC2019的数据加载逻辑
        split_file = self.data_root / f"{self.split}.txt"
        
        samples = []
        if split_file.exists():
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 假设格式：image_path depth_path
                        parts = line.split()
                        if len(parts) >= 2:
                            samples.append({
                                'image_path': str(self.data_root / parts[0]),
                                'depth_path': str(self.data_root / parts[1])
                            })
        else:
            # 直接从目录结构推断
            image_dir = self.data_root / "images"
            depth_dir = self.data_root / "heights"
            
            for image_path in sorted(image_dir.glob("*.tif")):
                depth_path = depth_dir / f"{image_path.stem}.tif"
                if depth_path.exists():
                    samples.append({
                        'image_path': str(image_path),
                        'depth_path': str(depth_path)
                    })
        
        print(f"Loaded {len(samples)} samples from DFC2019 {self.split}")
        return samples

class VaihingenDataset(BaseDepthDataset):
    """Vaihingen数据集"""
    
    def _load_samples(self) -> list:
        # Vaihingen的数据加载逻辑
        samples = []
        
        # 假设已经预处理成512x512的patches
        image_dir = self.data_root / self.split / "images"
        depth_dir = self.data_root / self.split / "depths"
        
        for image_path in sorted(image_dir.glob("*.png")):
            depth_path = depth_dir / f"{image_path.stem}.png"
            if depth_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'depth_path': str(depth_path)
                })
        
        print(f"Loaded {len(samples)} samples from Vaihingen {self.split}")
        return samples
class GoogleHeightDataset(BaseDepthDataset):
    """Google Height数据集"""
    
    def _load_samples(self) -> list:
        # GoogleHeightData的数据加载逻辑
        split_dir = self.data_root / self.split
        
        # 数据结构：GoogleHeightData/train/images/ 和 GoogleHeightData/train/labels/
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        
        samples = []
        for image_path in sorted(image_dir.glob("*.tif")):
            # 对应的深度文件应该有相同的文件名
            label_path = label_dir / image_path.name
            
            if label_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'depth_path': str(label_path)
                })
        
        print(f"Loaded {len(samples)} samples from GoogleHeightData {self.split}")
        return samples
def get_dataset(dataset_name: str, **kwargs) -> BaseDepthDataset:
    """工厂函数：根据名称获取数据集"""
    datasets = {
        'GAMUS': GAMUSDataset,
        'DFC2019': DFC2019Dataset, 
        'Vaihingen': VaihingenDataset,
        'GoogleHeight':GoogleHeightDataset,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")
    
    return datasets[dataset_name](**kwargs)