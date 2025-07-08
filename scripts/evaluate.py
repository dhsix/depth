#!/usr/bin/env python3
"""
评估脚本
用法:
    python scripts/evaluate.py --model_path checkpoints/best_model.pth --dataset GAMUS
    python scripts/evaluate.py --config configs/depth2elevation.yaml --checkpoint best
"""

import argparse
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configs.depth2elevation_config import Depth2ElevationConfig
from data.data_loader import create_data_loaders
from models import create_model
from utils.metrics import evaluate_depth_estimation, MetricsTracker
from utils.checkpoint import CheckpointManager
from utils.visualization import save_prediction_samples

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate depth estimation models')
    
    # 模型和配置
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--checkpoint', type=str, default='best',
                       choices=['best', 'latest'], help='Which checkpoint to use')
    
    # 数据集
    parser.add_argument('--dataset', type=str, default='GAMUS',
                       choices=['GAMUS', 'DFC2019', 'Vaihingen'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='/data/remote_sensing_datasets',
                       help='Data root directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'], help='Dataset split')
    
    # 评估选项
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    # 输出选项
    parser.add_argument('--save_predictions', action='store_true', help='Save prediction samples')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to save')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory')
    
    return parser.parse_args()

def load_model_and_config(args):
    """加载模型和配置"""
    
    if args.config:
        # 从配置文件加载
        from scripts.train import load_config_from_file
        config = load_config_from_file(args.config)
        
        # 检查点管理器
        checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        if args.checkpoint == 'best':
            checkpoint_info = checkpoint_manager.get_best_checkpoint()
        else:
            checkpoint_info = checkpoint_manager.get_latest_checkpoint()
        
        if not checkpoint_info:
            raise ValueError(f"No {args.checkpoint} checkpoint found in {config.checkpoint_dir}")
        
        model_path = checkpoint_info['path']
        
    elif args.model_path:
        # 直接从模型路径加载
        model_path = args.model_path
        
        # 尝试从检查点中恢复配置
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            if config_dict.get('model_name') == 'depth2elevation':
                config = Depth2ElevationConfig(**config_dict)
            else:
                from configs.base_config import BaseConfig
                config = BaseConfig(**config_dict)
        else:
            # 使用默认配置
            config = Depth2ElevationConfig()
            config.dataset_name = args.dataset
            config.data_root = args.data_root
    
    else:
        raise ValueError("Either --config or --model_path must be provided")
    
    # 更新数据集配置
    config.dataset_name = args.dataset
    config.data_root = args.data_root
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    
    # 创建模型
    model = create_model(config.to_dict())
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Model validation loss: {checkpoint.get('loss', 'unknown')}")
    
    return model, config

def evaluate_model(model, data_loader, device, save_dir=None, num_samples=0):
    """评估模型"""
    
    metrics_tracker = MetricsTracker()
    all_predictions = []
    all_targets = []
    sample_count = 0
    
    print(f"Evaluating on {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            # 移动数据到设备
            images = batch['image'].to(device)
            targets = batch['depth'].to(device)
            masks = batch.get('mask')
            if masks is not None:
                masks = masks.to(device)
            
            # 模型预测
            if hasattr(model, 'predict'):
                predictions = model.predict(images)
            else:
                predictions = model(images)
            
            # 处理多尺度输出
            if isinstance(predictions, dict):
                predictions = predictions.get('scale_4', list(predictions.values())[-1])
            
            # 计算指标
            try:
                batch_metrics = evaluate_depth_estimation(predictions, targets, masks)
                metrics_tracker.update(batch_metrics, batch_size=images.shape[0])
            except Exception as e:
                print(f"Warning: Failed to compute metrics for batch {batch_idx}: {e}")
                continue
            
            # 保存预测样本
            if save_dir and sample_count < num_samples:
                from utils.visualization import DepthVisualizer
                
                samples_to_save = min(images.shape[0], num_samples - sample_count)
                
                for i in range(samples_to_save):
                    sample_save_dir = Path(save_dir) / f"sample_{sample_count + i + 1}"
                    sample_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存可视化
                    fig = DepthVisualizer.visualize_prediction_batch(
                        images[i:i+1], predictions[i:i+1], targets[i:i+1], max_samples=1
                    )
                    fig.savefig(sample_save_dir / "comparison.png", dpi=150, bbox_inches='tight')
                    
                    # 保存原始数据
                    np.save(sample_save_dir / "prediction.npy", predictions[i].cpu().numpy())
                    np.save(sample_save_dir / "target.npy", targets[i].cpu().numpy())
                    if masks is not None:
                        np.save(sample_save_dir / "mask.npy", masks[i].cpu().numpy())
                
                sample_count += samples_to_save
            
            # 收集所有预测和目标（用于后续分析）
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # 计算最终指标
    final_metrics = metrics_tracker.compute()
    
    return final_metrics, torch.cat(all_predictions), torch.cat(all_targets)

def main():
    """主函数"""
    args = parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载模型和配置
        print("Loading model and config...")
        model, config = load_model_and_config(args)
        
        # 创建数据加载器
        print(f"Creating data loader for {args.split} split...")
        
        # 临时修改配置以只加载指定的split
        original_split = config.dataset_name
        
        if args.split == 'train':
            train_loader, _, _ = create_data_loaders(config)
            eval_loader = train_loader
        elif args.split == 'val':
            _, val_loader, _ = create_data_loaders(config)
            eval_loader = val_loader
        else:  # test
            _, _, test_loader = create_data_loaders(config)
            if test_loader is None:
                print("Test split not available, using validation split")
                _, eval_loader, _ = create_data_loaders(config)
            else:
                eval_loader = test_loader
        
        # 评估模型
        save_dir = output_dir / "samples" if args.save_predictions else None
        metrics, predictions, targets = evaluate_model(
            model, eval_loader, args.device, save_dir, args.num_samples
        )
        
        # 打印结果
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # 保存结果
        results_file = output_dir / f"evaluation_results_{args.split}.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # 保存详细结果为CSV
        results_df = pd.DataFrame([metrics])
        csv_file = output_dir / f"evaluation_results_{args.split}.csv"
        results_df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {results_file}")
        print(f"  CSV: {csv_file}")
        
        if args.save_predictions:
            print(f"  Samples: {save_dir}")
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()