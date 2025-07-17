import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import time
from abc import ABC, abstractmethod

from utils.logger import TrainingLogger
from utils.metrics import MetricsTracker, evaluate_depth_estimation
from utils.checkpoint import CheckpointManager
from utils.visualization import DepthVisualizer, save_prediction_samples

class BaseTrainer(ABC):
    """基础训练器抽象类"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # 训练参数
        self.num_epochs = config.get('num_epochs', 50)
        self.val_interval = config.get('val_interval', 5)
        self.save_interval = config.get('save_interval', 10)
        self.gradient_clip_val = config.get('gradient_clip_val', None)
        
        # 设置日志记录器
        self.logger = TrainingLogger(
            log_dir=config.get('log_dir', './logs'),
            experiment_name=config.get('experiment_name', 'experiment')
        )
        
        # 设置检查点管理器
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
            max_keep=config.get('max_keep_ckpts', 3),
            save_best=True
        )
        
        # 指标追踪器
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 记录配置
        self.logger.log_config(config)
        
        # 记录模型信息
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
            self.logger.log_info(f"Model Info: {model_info}")
    
    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练，子类必须实现"""
        pass
    
    @abstractmethod
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步验证，子类必须实现"""
        pass
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            batch = self._move_batch_to_device(batch)
            
            # 训练步骤
            step_metrics = self.train_step(batch)
            
            # 更新指标
            batch_size = batch['image'].shape[0]
            self.train_metrics.update(step_metrics, batch_size)
            
            # 梯度裁剪
            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            # 优化器步骤
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            # 记录训练进度
            if batch_idx % 100 == 0:
                current_metrics = self.train_metrics.compute()
                self.logger.log_info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {current_metrics.get('total_loss', 0):.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
        
        # 学习率调度
        if self.scheduler:
            self.scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        epoch_metrics = self.train_metrics.compute()
        epoch_metrics['epoch_time'] = epoch_time
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移动数据到设备
                batch = self._move_batch_to_device(batch)
                
                # 验证步骤
                step_metrics = self.val_step(batch)
                
                # 更新指标
                batch_size = batch['image'].shape[0]
                self.val_metrics.update(step_metrics, batch_size)
        
        return self.val_metrics.compute()
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """完整的训练流程"""
        
        # 恢复训练（如果指定）
        if resume_from:
            self._resume_training(resume_from)
        
        self.logger.log_info("Starting training...")
        self.logger.log_info(f"Total epochs: {self.num_epochs}")
        self.logger.log_info(f"Train batches: {len(self.train_loader)}")
        self.logger.log_info(f"Val batches: {len(self.val_loader)}")
        
        training_start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_metrics = self.train_epoch()
            
            # 验证阶段
            if epoch % self.val_interval == 0 or epoch == self.num_epochs - 1:
                val_metrics = self.validate_epoch()
                # 【修复】过滤掉可能冲突的键
                filtered_train_metrics = {
                    k: v for k, v in train_metrics.items() 
                    if k not in ['total_loss', 'loss']  # 过滤掉会冲突的键
                }
                filtered_val_metrics = {
                    k: v for k, v in val_metrics.items() 
                    if k not in ['total_loss', 'loss']  # 过滤掉会冲突的键
                }
                # # 记录epoch结果
                # self.logger.log_epoch(
                #     epoch=epoch,
                #     train_loss=train_metrics.get('total_loss', 0),
                #     val_loss=val_metrics.get('total_loss', 0),
                #     **{f"train_{k}": v for k, v in train_metrics.items() if k != 'total_loss'},
                #     **{f"val_{k}": v for k, v in val_metrics.items() if k != 'total_loss'}
                # )
                # 记录epoch结果
                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_metrics.get('total_loss', train_metrics.get('loss', 0)),
                    val_loss=val_metrics.get('total_loss', val_metrics.get('loss', 0)),
                    **{f"train_{k}": v for k, v in filtered_train_metrics.items()},
                    **{f"val_{k}": v for k, v in filtered_val_metrics.items()}
                )
                
                # 保存检查点
                is_best = self.checkpoint_manager.is_best_model(val_metrics.get('total_loss', float('inf')))
                
                if epoch % self.save_interval == 0 or is_best or epoch == self.num_epochs - 1:
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        loss=val_metrics.get('total_loss', 0),
                        metrics=val_metrics,
                        config=self.config,
                        is_best=is_best
                    )
                
                # 保存可视化样本
                if is_best:
                    self._save_visualization_samples(epoch)
            
            else:
                # 只记录训练指标
                filtered_train_metrics = {
                    k: v for k, v in train_metrics.items() 
                    if k not in ['total_loss', 'loss']
                }
                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_metrics.get('total_loss', train_metrics.get('loss', 0)),
                    val_loss=float('nan'),
                    **{f"train_{k}": v for k, v in filtered_train_metrics.items()}
                )
                # self.logger.log_epoch(
                #     epoch=epoch,
                #     train_loss=train_metrics.get('total_loss', 0),
                #     val_loss=float('nan'),
                #     **{f"train_{k}": v for k, v in train_metrics.items() if k != 'total_loss'}
                # )
        
        training_time = time.time() - training_start_time
        self.logger.log_info(f"Training completed in {training_time/3600:.2f} hours")
        
        # 保存训练指标
        self.logger.save_metrics()
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.num_epochs,
            'training_time': training_time
        }
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将batch移动到指定设备"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _resume_training(self, checkpoint_path: str):
        """恢复训练"""
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, self.device
        )
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('loss', float('inf'))
        
        self.logger.log_info(f"Resumed training from epoch {checkpoint['epoch']}")
    
    def _save_visualization_samples(self, epoch: int):
        """保存可视化样本"""
        try:
            save_dir = f"{self.config.get('result_dir', './results')}/samples_epoch_{epoch}"
            save_prediction_samples(self.model, self.val_loader, save_dir, num_samples=5)
        except Exception as e:
            self.logger.log_info(f"Warning: Failed to save visualization samples: {e}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        try:
            save_path = f"{self.config.get('result_dir', './results')}/training_curves.png"
            fig = DepthVisualizer.plot_training_curves(
                self.logger.metrics_history, save_path
            )
            if fig:
                self.logger.log_info(f"Training curves saved to {save_path}")
        except Exception as e:
            self.logger.log_info(f"Warning: Failed to plot training curves: {e}")