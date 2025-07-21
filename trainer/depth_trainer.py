import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .base_trainer import BaseTrainer
from utils.metrics import evaluate_depth_estimation

class DepthEstimationTrainer(BaseTrainer):
    """深度估计专用训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        
        super().__init__(model, train_loader, val_loader, optimizer, scheduler, config, device)
        
        # 深度估计特定参数
        self.use_mask = config.get('use_mask', True)
        self.compute_metrics_interval = config.get('compute_metrics_interval', 1)
        self._log_depth_estimation_setup()
    def _log_depth_estimation_setup(self):
        """记录深度估计特定的设置信息"""
        self.logger.log_info("🏗️ Depth Estimation Training Setup:")
        self.logger.log_info(f"   📏 Use Mask: {self.use_mask}")
        self.logger.log_info(f"   📊 Metrics Computation Interval: {self.compute_metrics_interval}")
        self.logger.log_info(f"   🎯 Multi-scale Output: {getattr(self.model, 'use_multi_scale_output', False)}")
        
        if hasattr(self.model, 'loss_fn'):
            self.logger.log_info(f"   📉 Loss Function: {type(self.model.loss_fn).__name__}")
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        
        images = batch['image']
        targets = batch['depth']
        masks = batch.get('mask') 
        
        # 前向传播
        if hasattr(self.model, 'forward'):
            # 检查模型是否支持多尺度输出
            if hasattr(self.model, 'use_multi_scale_output') and self.model.use_multi_scale_output:
                predictions = self.model(images, return_multi_scale=True)
            else:
                predictions = self.model(images)
        else:
            predictions = self.model(images)
        
        # 计算损失
        if hasattr(self.model, 'compute_loss'):
            loss_dict = self.model.compute_loss(predictions, targets, masks)
        else:
            # 使用简单的MSE损失作为后备
            if isinstance(predictions, dict):
                # 多尺度输出，只使用最高分辨率计算损失
                pred_tensor = predictions.get('scale_4', list(predictions.values())[-1])
            else:
                pred_tensor = predictions
            
            if masks is not None:
                valid_mask = masks.bool()
                loss = nn.functional.mse_loss(pred_tensor[valid_mask], targets[valid_mask])
            else:
                loss = nn.functional.mse_loss(pred_tensor, targets)
            
            loss_dict = {'total_loss': loss}
        
        # 反向传播
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        # 准备返回的指标
        step_metrics = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                step_metrics[key] = value.item()
            else:
                step_metrics[key] = value
        
        return step_metrics
    
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步验证"""
        
        images = batch['image']
        targets = batch['depth']
        masks = batch.get('mask') if self.use_mask else None
        
        # 前向传播
        if hasattr(self.model, 'predict'):
            if hasattr(self.model, 'use_multi_scale_output') and self.model.use_multi_scale_output:
                predictions = self.model.predict(images, return_multi_scale=True)
            else:
                predictions = self.model.predict(images)
        else:
            predictions = self.model(images)
        
        # 计算损失
        if hasattr(self.model, 'compute_loss'):
            loss_dict = self.model.compute_loss(predictions, targets, masks)
        else:
            # 后备损失计算
            if isinstance(predictions, dict):
                pred_tensor = predictions.get('scale_4', list(predictions.values())[-1])
            else:
                pred_tensor = predictions
            
            if masks is not None:
                valid_mask = masks.bool()
                loss = nn.functional.mse_loss(pred_tensor[valid_mask], targets[valid_mask])
            else:
                loss = nn.functional.mse_loss(pred_tensor, targets)
            
            loss_dict = {'total_loss': loss}
        
        # 计算评估指标
        step_metrics = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                step_metrics[key] = value.item()
            else:
                step_metrics[key] = value
        
        # 计算深度估计指标（每隔一定间隔计算，因为比较耗时）
        if self.current_epoch % self.compute_metrics_interval == 0:
            if isinstance(predictions, dict):
                pred_for_metrics = predictions.get('scale_4', list(predictions.values())[-1])
            else:
                pred_for_metrics = predictions
            
            try:
                depth_metrics = evaluate_depth_estimation(pred_for_metrics, targets, masks)
                step_metrics.update(depth_metrics)
                # 添加这些额外统计信息
                step_metrics['valid_pixels'] = masks.sum().item() if masks is not None else targets.numel()
                step_metrics['total_pixels'] = targets.numel()
                step_metrics['height_range'] = (targets.max() - targets.min()).item()
                step_metrics['mean_height'] = targets.mean().item()
            except Exception as e:
                # self.logger.log_info(f"Warning: Failed to compute depth metrics: {e}")
                self.logger.log_warning(f"Failed to compute depth metrics: {e}")
        
        return step_metrics

class MultiScaleDepthTrainer(DepthEstimationTrainer):
    """多尺度深度估计训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        
        super().__init__(model, train_loader, val_loader, optimizer, scheduler, config, device)
        
        # 多尺度特定参数
        self.scale_weights = config.get('scale_weights', [1.0, 1.0, 1.0, 1.0])
        self.log_scale_losses = config.get('log_scale_losses', True)
        # ✅ 添加这部分
        self.logger.log_info("🔄 Multi-Scale Configuration:")
        self.logger.log_info(f"   ⚖️  Scale Weights: {self.scale_weights}")
        self.logger.log_info(f"   📊 Log Scale Losses: {self.log_scale_losses}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """多尺度训练步骤"""
        
        images = batch['image']
        targets = batch['depth']
        masks = batch.get('mask') if self.use_mask else None
        
        # 强制使用多尺度输出
        predictions = self.model(images, return_multi_scale=True)
        
        # 计算多尺度损失
        loss_dict = self.model.compute_loss(predictions, targets, masks)
        
        # 反向传播
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        # 准备返回的指标
        step_metrics = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                step_metrics[key] = value.item()
            else:
                step_metrics[key] = value
        
        # 记录各尺度的损失（如果启用）
        if self.log_scale_losses and isinstance(predictions, dict):
            for scale_key, scale_pred in predictions.items():
                try:
                    # 计算单尺度损失用于监控
                    target_resized = torch.nn.functional.interpolate(
                        targets.unsqueeze(1),
                        size=scale_pred.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(1)
                    
                    scale_loss = nn.functional.mse_loss(scale_pred, target_resized)
                    step_metrics[f'loss_{scale_key}'] = scale_loss.item()
                except Exception as e:
                    self.logger.log_warning(f"Failed to compute scale loss for {scale_key}: {e}")
                # except:
                #     pass
        
        return step_metrics
    
    def val_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """多尺度验证步骤"""
        
        images = batch['image']
        targets = batch['depth']
        masks = batch.get('mask') if self.use_mask else None
        
        # 获取多尺度预测
        predictions = self.model.predict(images, return_multi_scale=True)
        
        # 计算损失
        loss_dict = self.model.compute_loss(predictions, targets, masks)
        
        step_metrics = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                step_metrics[key] = value.item()
            else:
                step_metrics[key] = value
        
        # 计算最高分辨率的深度指标
        if self.current_epoch % self.compute_metrics_interval == 0:
            pred_for_metrics = predictions.get('scale_4', list(predictions.values())[-1])
            
            try:
                depth_metrics = evaluate_depth_estimation(pred_for_metrics, targets, masks)
                step_metrics.update(depth_metrics)
            except Exception as e:
                # self.logger.log_info(f"Warning: Failed to compute depth metrics: {e}")
                self.logger.log_warning(f"Failed to compute depth metrics: {e}")
        
        # 记录各尺度损失
        if self.log_scale_losses:
            for scale_key, scale_pred in predictions.items():
                try:
                    target_resized = torch.nn.functional.interpolate(
                        targets.unsqueeze(1),
                        size=scale_pred.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(1)
                    
                    scale_loss = nn.functional.mse_loss(scale_pred, target_resized)
                    step_metrics[f'val_loss_{scale_key}'] = scale_loss.item()
                except Exception as e:
                    self.logger.log_warning(f"Failed to compute scale loss for {scale_key}: {e}")
                # except:
                #     pass

        
        return step_metrics

def create_trainer(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  optimizer: torch.optim.Optimizer,
                  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                  config: Dict[str, Any],
                  device: str = 'cuda') -> BaseTrainer:
    """训练器工厂函数"""
    
    trainer_type = config.get('trainer_type', 'depth')
    
    if trainer_type == 'multi_scale_depth':
        return MultiScaleDepthTrainer(
            model, train_loader, val_loader, optimizer, scheduler, config, device
        )
    elif trainer_type == 'depth':
        return DepthEstimationTrainer(
            model, train_loader, val_loader, optimizer, scheduler, config, device
        )
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")