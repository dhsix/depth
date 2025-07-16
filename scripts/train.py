#!/usr/bin/env python3
"""
主训练脚本
用法:
    python scripts/train.py --config configs/depth2elevation_gamus.yaml
    python scripts/train.py --model depth2elevation --dataset GAMUS --epochs 50
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from configs.base_config import BaseConfig
from configs.depth2elevation_config import Depth2ElevationConfig
from data.data_loader import create_data_loaders
from models import create_model
from trainer import create_trainer
from utils.optimizers import create_optimizer, create_scheduler

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train depth estimation models')
    
    # 配置文件
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # 基本参数
    parser.add_argument('--model', type=str, default='depth2elevation',
                       choices=['depth2elevation', 'depth_anything_v2', 'htc_dc', 'imele'],
                       help='Model name')
    parser.add_argument('--dataset', type=str, default='GAMUS',
                       choices=['GAMUS', 'DFC2019', 'Vaihingen'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='/data/remote_sensing_datasets',
                       help='Data root directory')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    # 模型参数
    parser.add_argument('--encoder', type=str, default='vitb',
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Encoder type for Depth2Elevation')
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained weights')
    
    # 系统参数
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 实验管理
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--result_dir', type=str, default='./experiments', help='Result directory')
    
    # 训练选项
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder')
    parser.add_argument('--val_interval', type=int, default=5, help='Validation interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    
    return parser.parse_args()

def load_config_from_file(config_path: str) -> BaseConfig:
    """从文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 根据文件扩展名选择加载方式
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 根据模型类型创建相应的配置
        model_name = config_dict.get('model_name', 'depth2elevation')
        # if model_name == 'depth2elevation':
        if model_name in ['depth2elevation','depth2elevation_multiscale','depth2elevation_gra']:
            return Depth2ElevationConfig(**config_dict)
        else:
            return BaseConfig(**config_dict)
    
    elif config_path.endswith('.json'):
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        model_name = config_dict.get('model_name', 'depth2elevation')
        if model_name == 'depth2elevation':
            return Depth2ElevationConfig(**config_dict)
        else:
            return BaseConfig(**config_dict)
    
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")

def create_config_from_args(args) -> BaseConfig:
    """从命令行参数创建配置"""
    
    # 生成实验名称
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        experiment_name = f"{args.model}_{args.dataset}_{args.encoder}_{args.epochs}ep"
    
    # 基础配置
    config_dict = {
        'model_name': args.model,
        'dataset_name': args.dataset,
        'data_root': args.data_root,
        'experiment_name': experiment_name,
        
        # 训练参数
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'seed': args.seed,
        
        # 验证和保存
        'val_interval': args.val_interval,
        'save_interval': args.save_interval,
        
        # 路径
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'result_dir': args.result_dir,
        
        # 系统
        'device': args.device,
    }
    
    # 模型特定配置
    if args.model == 'depth2elevation':
        config_dict.update({
            'model_config': {
                'encoder': args.encoder,
                'pretrained_path': args.pretrained_path,
                'freeze_encoder': args.freeze_encoder,
            },
            'loss_config': {
                'type': 'multi_scale_loss',
                'gamma': 1.0,
                'delta': 1.0,
                'mu': 0.05,
            },
            'trainer_type': 'multi_scale_depth',
            'use_multi_scale_output': True,
        })
        return Depth2ElevationConfig(**config_dict)
    
    else:
        # 其他模型使用基础配置
        config_dict.update({
            'model_config': {
                'encoder': args.encoder,
                'pretrained_path': args.pretrained_path,
            },
            'loss_config': {
                'type': 'single_scale_loss',
                'criterion': 'mse',
            },
            'trainer_type': 'depth',
        })
        return BaseConfig(**config_dict)

def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    if args.config:
        config = load_config_from_file(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = create_config_from_args(args)
        print("Created config from command line arguments")
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 检查设备
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        config.device = 'cpu'
    
    print(f"Using device: {config.device}")
    print(f"Experiment: {config.experiment_name}")
    
    # 保存配置
    experiment_dir = Path(config.get_experiment_dir())
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(config, 'save_yaml'):
        config_save_path = experiment_dir / "config.yaml"
        config.save_yaml(str(config_save_path))
        print(f"Config saved to: {config_save_path}")
    
    try:
        # 创建数据加载器
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # 创建模型
        print(f"Creating model: {config.model_name}")
        model = create_model(config.to_dict())
        print(f"Model info: {model.get_model_info()}")
        
        # 创建优化器和调度器
        print("Creating optimizer and scheduler...")
        optimizer = create_optimizer(model.parameters(), config.to_dict())
        scheduler = create_scheduler(optimizer, config.to_dict())
        
        # 创建训练器
        print("Creating trainer...")
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config.to_dict(),
            device=config.device
        )
        
        # 开始训练
        print("Starting training...")
        results = trainer.train(resume_from=args.resume)
        
        print("Training completed successfully!")
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
        print(f"Total training time: {results['training_time']/3600:.2f} hours")
        
        # 保存最终结果
        results_file = experiment_dir / "final_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()