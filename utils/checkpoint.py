import torch
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

class CheckpointManager:
    """模型检查点管理器"""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 max_keep: int = 3,
                 save_best: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_keep = max_keep
        self.save_best = save_best
        
        self.best_metric = float('inf')
        self.saved_checkpoints = []
        
        # 检查点信息文件
        self.info_file = self.checkpoint_dir / 'checkpoint_info.json'
        self.load_checkpoint_info()
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       loss: float,
                       metrics: Dict[str, float],
                       config: Dict[str, Any],
                       is_best: bool = False) -> str:
        """保存检查点"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            checkpoint_name = f'best_model_epoch_{epoch}_{timestamp}.pth'
        else:
            checkpoint_name = f'checkpoint_epoch_{epoch}_{timestamp}.pth'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # 准备检查点数据
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'metrics': metrics,
            'config': config,
            'timestamp': timestamp,
            'is_best': is_best
        }
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        
        # 更新检查点列表
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'timestamp': timestamp,
            'is_best': is_best
        }
        
        self.saved_checkpoints.append(checkpoint_info)
        
        # 管理检查点数量
        if not is_best:
            self._manage_checkpoints()
        
        # 保存检查点信息
        self.save_checkpoint_info()
        
        print(f"Saved checkpoint: {checkpoint_name}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       device: str = 'cuda') -> Dict[str, Any]:
        """加载检查点"""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
        
        return checkpoint
    
    def load_best_checkpoint(self,
                           model: torch.nn.Module,
                           optimizer: Optional[torch.optim.Optimizer] = None,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                           device: str = 'cuda') -> Optional[Dict[str, Any]]:
        """加载最佳检查点"""
        
        best_checkpoint = self.get_best_checkpoint()
        if best_checkpoint:
            return self.load_checkpoint(
                best_checkpoint['path'], model, optimizer, scheduler, device
            )
        else:
            print("No best checkpoint found")
            return None
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """获取最佳检查点信息"""
        best_checkpoints = [cp for cp in self.saved_checkpoints if cp['is_best']]
        if best_checkpoints:
            # 返回损失最小的
            return min(best_checkpoints, key=lambda x: x['loss'])
        return None
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """获取最新检查点信息"""
        if self.saved_checkpoints:
            return max(self.saved_checkpoints, key=lambda x: x['epoch'])
        return None
    
    def _manage_checkpoints(self):
        """管理检查点数量"""
        # 只管理非best检查点
        non_best_checkpoints = [cp for cp in self.saved_checkpoints if not cp['is_best']]
        
        if len(non_best_checkpoints) > self.max_keep:
            # 按epoch排序，删除最旧的
            non_best_checkpoints.sort(key=lambda x: x['epoch'])
            to_remove = non_best_checkpoints[:-self.max_keep]
            
            for checkpoint_info in to_remove:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"Removed old checkpoint: {checkpoint_path.name}")
                
                # 从列表中移除
                self.saved_checkpoints.remove(checkpoint_info)
    
    def save_checkpoint_info(self):
        """保存检查点信息"""
        with open(self.info_file, 'w') as f:
            json.dump({
                'best_metric': self.best_metric,
                'saved_checkpoints': self.saved_checkpoints
            }, f, indent=2)
    
    def load_checkpoint_info(self):
        """加载检查点信息"""
        if self.info_file.exists():
            try:
                with open(self.info_file, 'r') as f:
                    info = json.load(f)
                    self.best_metric = info.get('best_metric', float('inf'))
                    self.saved_checkpoints = info.get('saved_checkpoints', [])
            except Exception as e:
                print(f"Warning: Failed to load checkpoint info: {e}")
                self.best_metric = float('inf')
                self.saved_checkpoints = []
    
    def is_best_model(self, current_metric: float) -> bool:
        """判断是否是最佳模型"""
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            return True
        return False
    
    def cleanup_old_checkpoints(self, keep_best: bool = True):
        """清理旧的检查点"""
        for checkpoint_info in self.saved_checkpoints.copy():
            if not keep_best or not checkpoint_info['is_best']:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                self.saved_checkpoints.remove(checkpoint_info)
        
        self.save_checkpoint_info()
        print("Cleaned up old checkpoints")