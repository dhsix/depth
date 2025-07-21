import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Dict, Any, Tuple, Optional
from .datasets import get_dataset
from .transforms import get_transforms

def collate_fn(batch):
    """自定义collate函数"""
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        if key in ['image', 'depth', 'mask','semantic_mask']:
            # 数值数据，直接stack
            result[key] = torch.stack([item[key] for item in batch])
        else:
            # 路径等元数据，保持为列表
            result[key] = [item[key] for item in batch]
    
    return result

def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """创建训练、验证和测试数据加载器"""
    # 🔥 从配置中获取是否加载mask
    load_mask = getattr(config, 'load_mask', True)
    # 获取数据变换
    train_transform = get_transforms(
        {
            'input_size': config.input_size,
            'patch_size': config.patch_size,
            **config.augmentation_config
        }, 
        is_training=True
    )
    
    val_transform = get_transforms(
        {
            'input_size': config.input_size,
            'patch_size': config.patch_size
        }, 
        is_training=False
    )
    
    # 获取完整的数据集路径

    dataset_path=config.data_root

    
    # 创建数据集
    train_dataset = get_dataset(
        dataset_name=config.dataset_name,
        data_root=dataset_path,  # 传入完整路径
        split='train',
        transform=train_transform,
        load_mask=load_mask
    )
    
    val_dataset = get_dataset(
        dataset_name=config.dataset_name,
        data_root=dataset_path,  # 传入完整路径
        split='val',
        transform=val_transform,
        load_mask=load_mask
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=SequentialSampler(val_dataset),
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    # 可选的测试集
    test_loader = None
    try:
        test_dataset = get_dataset(
            dataset_name=config.dataset_name,
            data_root=dataset_path,  # 使用完整路径
            split='test',
            transform=val_transform,
            load_mask=load_mask
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            sampler=SequentialSampler(test_dataset),
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False
        )
    except:
        print("No test dataset found, skipping...")
    
    print(f"Created data loaders:")
    print(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
    print(f"  Val: {len(val_loader)} batches, {len(val_dataset)} samples")
    if test_loader:
        print(f"  Test: {len(test_loader)} batches, {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def get_sample_batch(config, split: str = 'train') -> Dict[str, Any]:
    """获取一个样本batch，用于调试"""
    transform = get_transforms(
        {
            'input_size': config.input_size,
            'patch_size': config.patch_size
        }, 
        is_training=False
    )
    load_mask = getattr(config, 'load_mask', True)
    dataset = get_dataset(
        dataset_name=config.dataset_name,
        data_root=config.data_root,
        split=split,
        transform=transform,
        load_mask=load_mask
    )
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    return next(iter(loader))