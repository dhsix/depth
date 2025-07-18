import torch
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error

class DepthMetrics:
    """深度估计评估指标"""
    
    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Mean Absolute Error"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return float('inf')
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        return torch.mean(torch.abs(pred_valid - target_valid)).item()
    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Root Mean Square Error"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return float('inf')
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        return torch.sqrt(torch.mean((pred_valid - target_valid) ** 2)).item()
    @staticmethod
    def si_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Scale-Invariant Root Mean Square Error - 按论文公式"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return float('inf')
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # SI_RMSE = RMSE / σy
        rmse = torch.sqrt(torch.mean((pred_valid - target_valid) ** 2))
        target_std = torch.std(target_valid)
        si_rmse = rmse / target_std if target_std > 1e-8 else rmse
        
        return si_rmse.item()
    @staticmethod
    def logsi_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Log-based Scale-Invariant Root Mean Square Error"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return float('inf')
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # 避免log(0)
        epsilon = 1e-7
        log_pred = torch.log(pred_valid + epsilon)
        log_target = torch.log(target_valid + epsilon)
        
        diff = log_pred - log_target
        logsi_rmse = torch.sqrt(torch.mean(diff ** 2) - (torch.mean(diff) ** 2))
        
        return logsi_rmse.item()
    
    @staticmethod
    def abs_rel(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Absolute Relative Error"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return float('inf')
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # 避免除零
        target_valid = torch.clamp(target_valid, min=1e-8)
        abs_rel = torch.mean(torch.abs(pred_valid - target_valid) / target_valid)
        
        return abs_rel.item()
    
    @staticmethod
    def sq_rel(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Squared Relative Error"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return float('inf')
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # 避免除零
        target_valid = torch.clamp(target_valid, min=1e-8)
        sq_rel = torch.mean(((pred_valid - target_valid) ** 2) / target_valid)
        
        return sq_rel.item()
    
    @staticmethod
    def delta_threshold(pred: torch.Tensor, target: torch.Tensor, 
                       threshold: float = 1.25, mask: Optional[torch.Tensor] = None) -> float:
        """Delta threshold accuracy"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return 0.0
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # 避免除零
        pred_valid = torch.clamp(pred_valid, min=1e-8)
        target_valid = torch.clamp(target_valid, min=1e-8)
        
        ratio = torch.max(pred_valid / target_valid, target_valid / pred_valid)
        accuracy = torch.mean((ratio < threshold).float())
        
        return accuracy.item()

def evaluate_depth_estimation(pred: torch.Tensor, 
                            target: torch.Tensor,
                            mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """计算所有深度估计指标"""
    
    # 确保输入是同样的形状
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    
    # 转移到CPU并转换为numpy（如果需要）
    if pred.is_cuda:
        pred = pred.cpu()
    if target.is_cuda:
        target = target.cpu()
    if mask is not None and mask.is_cuda:
        mask = mask.cpu()
    
    metrics = {
        'MAE': DepthMetrics.mae(pred, target, mask),
        'RMSE': DepthMetrics.rmse(pred, target, mask),
        'SI_RMSE': DepthMetrics.si_rmse(pred, target, mask),
        'LOGSI_RMSE': DepthMetrics.logsi_rmse(pred, target, mask),
        'AbsRel': DepthMetrics.abs_rel(pred, target, mask),
        'SqRel': DepthMetrics.sq_rel(pred, target, mask),
        'δ<1.25': DepthMetrics.delta_threshold(pred, target, 1.25, mask),
        'δ<1.25²': DepthMetrics.delta_threshold(pred, target, 1.25**2, mask),
        'δ<1.25³': DepthMetrics.delta_threshold(pred, target, 1.25**3, mask),
    }
    
    return metrics

class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        """更新指标"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            # 只处理数值类型的指标，跳过字典、列表等复杂类型
            if isinstance(value, (int, float)):
                self.metrics[key] += value * batch_size
                self.counts[key] += batch_size
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                # 处理单元素张量
                self.metrics[key] += value.item() * batch_size
                self.counts[key] += batch_size
            else:
                # 对于复杂类型（dict、list等），直接存储最新值，不累计
                self.metrics[key] = value
                self.counts[key] = 1  # 设置为1，避免除零

            # self.metrics[key] += value * batch_size
            # self.counts[key] += batch_size
    
    def compute(self) -> Dict[str, float]:
        """计算平均指标"""
        averaged_metrics = {}
        for key in self.metrics:
            # if self.counts[key] > 0:
            #     averaged_metrics[key] = self.metrics[key] / self.counts[key]
            # else:
            #     averaged_metrics[key] = 0.0

            if self.counts[key] > 0:
                # 只对数值类型计算平均值
                if isinstance(self.metrics[key], (int, float)):
                    averaged_metrics[key] = self.metrics[key] / self.counts[key]
                else:
                    # 对于复杂类型，直接返回最新值
                    averaged_metrics[key] = self.metrics[key]
            else:
                averaged_metrics[key] = 0.0
        
        return averaged_metrics
    
    def get_summary(self) -> str:
        """获取指标摘要"""
        metrics = self.compute()
        summary_lines = []
        
        # 主要指标
        main_metrics = ['MAE', 'RMSE', 'SI_RMSE','LOGSI_RMSE']
        summary_lines.append("Main Metrics:")
        for metric in main_metrics:
            if metric in metrics:
                summary_lines.append(f"  {metric}: {metrics[metric]:.4f}")
        
        # 相对误差
        rel_metrics = ['AbsRel', 'SqRel']
        if any(m in metrics for m in rel_metrics):
            summary_lines.append("Relative Errors:")
            for metric in rel_metrics:
                if metric in metrics:
                    summary_lines.append(f"  {metric}: {metrics[metric]:.4f}")
        
        # 准确率
        acc_metrics = [k for k in metrics.keys() if k.startswith('δ')]
        if acc_metrics:
            summary_lines.append("Threshold Accuracies:")
            for metric in acc_metrics:
                summary_lines.append(f"  {metric}: {metrics[metric]:.4f}")
        
        return "\n".join(summary_lines)