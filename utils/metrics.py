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
                print('valid_mask_sum 0')
                return 0
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        # 移除nan和inf
        valid_indices = torch.isfinite(pred_valid) & torch.isfinite(target_valid)
        if valid_indices.sum() == 0:
            print('valid_indices_sum 0')
            return 0.0
        pred_clean = pred_valid[valid_indices]
        target_clean = target_valid[valid_indices]
        mae = torch.mean(torch.abs(pred_clean - target_clean)).item()
        return mae if np.isfinite(mae) else 0.0

    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Root Mean Square Error"""
        """Root Mean Square Error"""
        try:
            if mask is not None:
                valid_mask = mask.bool()
                if valid_mask.sum() == 0:
                    return 0.0
                pred_valid = pred[valid_mask]
                target_valid = target[valid_mask]
            else:
                pred_valid = pred.flatten()
                target_valid = target.flatten()
            
            # 移除nan和inf
            valid_indices = torch.isfinite(pred_valid) & torch.isfinite(target_valid)
            if valid_indices.sum() == 0:
                return 0.0
            
            pred_clean = pred_valid[valid_indices]
            target_clean = target_valid[valid_indices]
            
            mse = torch.mean((pred_clean - target_clean) ** 2)
            rmse = torch.sqrt(mse).item()
            return rmse if np.isfinite(rmse) else 0.0
            
        except Exception as e:
            print(f"⚠️ RMSE计算错误: {e}")
            return 0.0
    @staticmethod
    def si_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Scale-Invariant Root Mean Square Error - 按论文公式"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return 0.0  # 🔥 修复：改为返回0.0
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # 🔥 添加安全检查
        valid_indices = torch.isfinite(pred_valid) & torch.isfinite(target_valid)
        if valid_indices.sum() == 0:
            return 0.0
        
        pred_clean = pred_valid[valid_indices]
        target_clean = target_valid[valid_indices]
        
        # SI_RMSE = RMSE / σy
        rmse = torch.sqrt(torch.mean((pred_clean - target_clean) ** 2))
        target_std = torch.std(target_clean)
        
        # 🔥 修复：避免除零
        if target_std < 1e-6:  # 如果标准差太小
            return rmse.item() if np.isfinite(rmse.item()) else 0.0
        
        si_rmse = rmse / target_std
        result = si_rmse.item()
        return result if np.isfinite(result) else 0.0
    @staticmethod
    def logsi_rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Log-based Scale-Invariant Root Mean Square Error"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return 0.0  # 🔥 修复：改为返回0.0
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # 🔥 修复：更严格的正值检查
        valid_indices = (torch.isfinite(pred_valid) & 
                        torch.isfinite(target_valid) & 
                        (pred_valid > 0.01) &  # 确保是正值，避免log(0)
                        (target_valid > 0.01))
        
        if valid_indices.sum() == 0:
            print('logsi_rmse: 没有有效的正值像素')
            return 0.0
        
        pred_clean = pred_valid[valid_indices]
        target_clean = target_valid[valid_indices]
        
        # 避免log(0) - 使用更大的epsilon
        epsilon = 1e-6  # 🔥 增大epsilon
        log_pred = torch.log(pred_clean + epsilon)
        log_target = torch.log(target_clean + epsilon)
        
        diff = log_pred - log_target
        
        # 🔥 检查diff是否合理
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            print('logsi_rmse: log差值包含NaN/inf')
            return 0.0
        
        if len(diff) < 2:  # 需要至少2个点计算方差
            return 0.0
        
        mean_diff = torch.mean(diff)
        var_diff = torch.mean(diff ** 2) - mean_diff ** 2
        
        if var_diff < 0:  # 数值误差导致负方差
            var_diff = torch.mean(diff ** 2)
        
        logsi_rmse = torch.sqrt(var_diff)
        result = logsi_rmse.item()
        return result if np.isfinite(result) else 0.0

    
    @staticmethod
    def abs_rel(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Absolute Relative Error"""
        """Absolute Relative Error"""
        try:
            if mask is not None:
                valid_mask = mask.bool()
                if valid_mask.sum() == 0:
                    return 0.0
                pred_valid = pred[valid_mask]
                target_valid = target[valid_mask]
            else:
                pred_valid = pred.flatten()
                target_valid = target.flatten()
            
            # 🔥 关键修复：更严格的有效性检查
            # 1. 移除nan和inf
            finite_mask = torch.isfinite(pred_valid) & torch.isfinite(target_valid)
            
            # 2. 移除零值和负值（对于高度数据，应该都是正值）
            positive_mask = (target_valid > 0.1)  # 设置最小阈值0.1米
            
            # 3. 组合掩码
            valid_mask_combined = finite_mask & positive_mask
            
            if valid_mask_combined.sum() == 0:
                print("⚠️ AbsRel: 没有有效像素")
                return 0.0
            
            pred_clean = pred_valid[valid_mask_combined]
            target_clean = target_valid[valid_mask_combined]
            
            # 4. 再次确保target不为零
            target_clean = torch.clamp(target_clean, min=0.1)
            
            # 5. 计算相对误差
            abs_rel = torch.mean(torch.abs(pred_clean - target_clean) / target_clean)
            
            result = abs_rel.item()
            # 6. 检查结果是否合理
            if not np.isfinite(result) or result > 100:  # 如果相对误差超过100倍，可能有问题
                print(f"⚠️ AbsRel异常值: {result}, 设为0")
                return 0.0
            
            return result
    
    @staticmethod
    def sq_rel(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Squared Relative Error"""
        """Squared Relative Error - 关键修复"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return 0.0  # 🔥 修复：改为返回0.0
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # 🔥 同abs_rel的严格检查
        valid_indices = (torch.isfinite(pred_valid) & 
                        torch.isfinite(target_valid) & 
                        (target_valid > 0.5))  # 🔥 提高最小阈值
        
        if valid_indices.sum() == 0:
            print('sq_rel: 没有有效像素（target > 0.5m）')
            return 0.0
        
        pred_clean = pred_valid[valid_indices]
        target_clean = target_valid[valid_indices]
        
        # 避免除零 - 🔥 使用更大的clamp值
        target_clean = torch.clamp(target_clean, min=0.5)
        sq_rel = torch.mean(((pred_clean - target_clean) ** 2) / target_clean)
        
        result = sq_rel.item()
        
        # 🔥 添加合理性检查
        if not np.isfinite(result) or result > 500:  # 平方项会更大，阈值设为500
            print(f'sq_rel异常值: {result}, 设为0')
            return 0.0
        
        return result
    
    @staticmethod
    def delta_threshold(pred: torch.Tensor, target: torch.Tensor, 
                       threshold: float = 1.25, mask: Optional[torch.Tensor] = None) -> float:
        """Delta threshold accuracy - 轻微修复"""
        if mask is not None:
            valid_mask = mask.bool()
            if valid_mask.sum() == 0:
                return 0.0
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
        else:
            pred_valid = pred.flatten()
            target_valid = target.flatten()
        
        # 🔥 使用更大的clamp值
        valid_indices = (torch.isfinite(pred_valid) & 
                        torch.isfinite(target_valid) & 
                        (pred_valid > 0.1) & 
                        (target_valid > 0.1))
        
        if valid_indices.sum() == 0:
            return 0.0
        
        pred_clean = pred_valid[valid_indices]
        target_clean = target_valid[valid_indices]
        
        # 避免除零
        pred_clean = torch.clamp(pred_clean, min=0.1)  # 🔥 改为0.1
        target_clean = torch.clamp(target_clean, min=0.1)
        
        ratio = torch.max(pred_clean / target_clean, target_clean / pred_clean)
        
        # 🔥 添加异常值检查
        ratio = torch.clamp(ratio, max=100)  # 限制最大比值
        
        accuracy = torch.mean((ratio < threshold).float())
        
        result = accuracy.item()
        return result if np.isfinite(result) else 0.0

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