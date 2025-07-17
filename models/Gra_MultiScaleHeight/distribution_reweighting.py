import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from scipy.ndimage import convolve1d

class MultiScaleHeightDistributionAnalyzer(nn.Module):
    """
    多尺度高度分布分析器 - 实现 Label Distribution Smoothing (LDS)
    
    基于 Yang et al. ICML 2021 "Delving into Deep Imbalanced Regression" 
    首次应用于遥感建筑高度估计的长尾分布问题
    """
    def __init__(self, 
                 num_height_bins: int = 50,
                 max_height: float = 100.0,
                 scales: List[str] = ['scale_1', 'scale_2', 'scale_3', 'scale_4'],
                 lds_kernel: str = 'gaussian',
                 lds_ks: int = 5,
                 lds_sigma: float = 2.0):
        super().__init__()
        
        self.num_height_bins = num_height_bins
        self.max_height = max_height
        self.scales = scales
        self.lds_kernel = lds_kernel
        self.lds_ks = lds_ks
        self.lds_sigma = lds_sigma
        
        # 高度区间划分
        self.register_buffer('height_bins', 
                           torch.linspace(0, max_height, num_height_bins + 1))
        
        # 每个尺度的动态权重预测器
        self.weight_predictors = nn.ModuleDict({
            scale: nn.Sequential(
                nn.Linear(num_height_bins, 64),
                nn.ReLU(True),
                nn.Linear(64, 32),
                nn.ReLU(True),
                nn.Linear(32, 1),
                nn.Softplus()  # 保证权重为正
            ) for scale in scales
        })
        
        # 分布统计追踪（移动平均）
        for scale in scales:
            self.register_buffer(f'running_dist_{scale}', 
                               torch.ones(num_height_bins) / num_height_bins)
        
        self.momentum = 0.9
        
        # 多尺度一致性权重
        self.consistency_weight = nn.Parameter(torch.tensor(0.1))
        
        # 预计算LDS核
        self.register_buffer('lds_kernel_window', 
                           torch.tensor(self._get_lds_kernel_window(), dtype=torch.float32))
    
    def _get_lds_kernel_window(self) -> np.ndarray:
        """
        生成LDS平滑核 - 基于ICML 2021论文实现
        """
        assert self.lds_kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (self.lds_ks - 1) // 2
        
        if self.lds_kernel == 'gaussian':
            # 高斯核
            from scipy.ndimage import gaussian_filter1d
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            kernel_window = gaussian_filter1d(base_kernel, sigma=self.lds_sigma)
            kernel_window = kernel_window / max(kernel_window)
        elif self.lds_kernel == 'triang':
            # 三角核
            from scipy.signal.windows import triang
            kernel_window = triang(self.lds_ks)
        else:  # laplace
            # 拉普拉斯核
            laplace = lambda x: np.exp(-abs(x) / self.lds_sigma) / (2. * self.lds_sigma)
            kernel_window = np.array([laplace(x) for x in range(-half_ks, half_ks + 1)])
            kernel_window = kernel_window / max(kernel_window)
        
        return kernel_window
    
    def compute_empirical_distribution(self, height_values: torch.Tensor) -> torch.Tensor:
        """
        计算经验高度分布
        
        Args:
            height_values: 高度值 [B, H, W] 或 [B*H*W]
        Returns:
            empirical_dist: 经验分布 [B, num_bins] 或 [num_bins]
        """
        if height_values.dim() == 3:
            # 处理批量数据 [B, H, W]
            B, H, W = height_values.shape
            batch_distributions = []
            
            for b in range(B):
                height_flat = height_values[b].flatten()
                # 计算直方图
                hist = torch.histc(height_flat, bins=self.num_height_bins, 
                                 min=0, max=self.max_height)

                hist_sum = hist.sum()
                if hist_sum > 1e-8:
                    density = hist / hist_sum
                else:
                    density = torch.ones_like(hist) / self.num_height_bins
                
                # # 归一化为概率密度
                # density = hist / (hist.sum() + 1e-8)
                batch_distributions.append(density)
            
            return torch.stack(batch_distributions, dim=0)
        else:
            # 处理单个分布
            height_flat = height_values.flatten()
            hist = torch.histc(height_flat, bins=self.num_height_bins,
                             min=0, max=self.max_height)
            hist_sum = hist.sum()
            if hist_sum > 1e-8:
                density = hist / hist_sum
            else:
                density = torch.ones_like(hist) / self.num_height_bins
            # density = hist / (hist.sum() + 1e-8)
            return density
    
    def apply_label_distribution_smoothing(self, empirical_dist: torch.Tensor) -> torch.Tensor:
        """
        应用Label Distribution Smoothing (LDS)
        
        Args:
            empirical_dist: 经验分布 [B, num_bins] 或 [num_bins]
        Returns:
            smoothed_dist: 平滑后的分布
        """
        if empirical_dist.dim() == 1:
            # 单个分布
            empirical_np = empirical_dist.cpu().numpy()
            smoothed_np = convolve1d(empirical_np, 
                                   weights=self.lds_kernel_window.cpu().numpy(), 
                                   mode='constant')
            return torch.tensor(smoothed_np, device=empirical_dist.device, dtype=empirical_dist.dtype)
        else:
            # 批量处理
            B = empirical_dist.shape[0]
            smoothed_distributions = []
            
            for b in range(B):
                empirical_np = empirical_dist[b].cpu().numpy()
                smoothed_np = convolve1d(empirical_np,
                                       weights=self.lds_kernel_window.cpu().numpy(),
                                       mode='constant')
                smoothed_tensor = torch.tensor(smoothed_np, 
                                             device=empirical_dist.device,
                                             dtype=empirical_dist.dtype)
                smoothed_distributions.append(smoothed_tensor)
            
            return torch.stack(smoothed_distributions, dim=0)
    
    def compute_effective_weights(self, 
                                smoothed_dist: torch.Tensor,
                                scale_name: str) -> torch.Tensor:
        """
        基于平滑分布计算有效权重
        
        结合 Effective Number 和动态调制
        """
        # 1. Effective Number 计算 (基于ICML 2021)
        beta = 0.999
        total_samples = 10000  # 估计的总样本数
        estimated_counts = smoothed_dist * total_samples
        
        effective_nums = (1.0 - torch.pow(beta, estimated_counts)) / (1.0 - beta + 1e-8)
        
        # 2. 基础重加权（逆频率）
        base_weights = 1.0 / (effective_nums + 1e-8)
        
        # 3. 高度困难度调制
        # 高建筑物（>20m）被视为困难样本
        height_centers = (self.height_bins[:-1] + self.height_bins[1:]) / 2
        difficulty_factor = torch.sigmoid((height_centers - 20.0) / 10.0)
        
        if smoothed_dist.dim() == 1:
            # 单个分布
            difficulty_weights = base_weights * difficulty_factor
        else:
            # 批量处理
            difficulty_weights = base_weights * difficulty_factor.unsqueeze(0)
        
        # 4. 动态调制
        if smoothed_dist.dim() == 1:
            dynamic_input = smoothed_dist.unsqueeze(0)
        else:
            dynamic_input = smoothed_dist
            
        dynamic_modulation = self.weight_predictors[scale_name](dynamic_input)  # [B, 1]
        
        if smoothed_dist.dim() == 1:
            dynamic_modulation = dynamic_modulation.squeeze(0)
            final_weights = difficulty_weights * dynamic_modulation.squeeze()
        else:
            final_weights = difficulty_weights * dynamic_modulation
        
        # 5. 归一化
        if final_weights.dim() == 1:
            final_weights = final_weights / (final_weights.mean() + 1e-8)
        else:
            final_weights = final_weights / (final_weights.mean(dim=1, keepdim=True) + 1e-8)
        
        return final_weights
    
    def map_weights_to_pixels(self,
                            predictions: torch.Tensor,
                            targets: torch.Tensor,
                            bin_weights: torch.Tensor) -> torch.Tensor:
        """
        将bin权重映射到像素权重
        
        Args:
            predictions: 预测结果 [B, H, W]
            targets: 真实标签 [B, H, W]  
            bin_weights: bin权重 [B, num_bins] 或 [num_bins]
        Returns:
            pixel_weights: 像素权重 [B, H, W]
        """
        # 调整目标大小以匹配预测
        if targets.shape != predictions.shape:
            targets_resized = F.interpolate(
                targets.unsqueeze(1).float(),
                size=predictions.shape[-2:],
                mode='bilinear',
                align_corners=True
            ).squeeze(1)
        else:
            targets_resized = targets
        
        # 计算每个像素属于哪个bin
        height_indices = torch.clamp(
            (targets_resized / self.max_height * self.num_height_bins).long(),
            0, self.num_height_bins - 1
        )
        
        B, H, W = targets_resized.shape
        pixel_weights = torch.zeros_like(targets_resized)
        
        if bin_weights.dim() == 1:
            # 所有批次使用相同权重
            for b in range(B):
                pixel_weights[b] = bin_weights[height_indices[b]]
        else:
            # 每个批次使用不同权重
            for b in range(B):
                pixel_weights[b] = bin_weights[b][height_indices[b]]
        
        return pixel_weights
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        主要前向传播函数
        
        Args:
            predictions: 多尺度预测结果 {'scale_1': [B,H,W], ...}
            targets: 真实高度图 [B, H, W]
        
        Returns:
            pixel_weights: 每个尺度的像素权重
            analysis_info: 分布分析信息
        """
        pixel_weights = {}
        scale_distributions = {}
        scale_weights = {}
        
        # 1. 对每个尺度进行分布分析
        for scale_name, pred in predictions.items():
            if scale_name not in self.scales:
                continue
            
            # 调整目标尺寸到预测尺寸
            if targets.shape != pred.shape:
                targets_resized = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=pred.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)
            else:
                targets_resized = targets
            
            # 计算经验分布
            empirical_dist = self.compute_empirical_distribution(targets_resized)
            scale_distributions[scale_name] = empirical_dist
            
            # 应用LDS平滑
            smoothed_dist = self.apply_label_distribution_smoothing(empirical_dist)
            
            # 计算有效权重
            bin_weights = self.compute_effective_weights(smoothed_dist, scale_name)
            scale_weights[scale_name] = bin_weights
            
            # 映射到像素权重
            pixel_weight_map = self.map_weights_to_pixels(pred, targets_resized, bin_weights)
            pixel_weights[scale_name] = pixel_weight_map
            
            # 更新移动平均分布（仅训练时）
            if self.training:
                if empirical_dist.dim() == 1:
                    global_dist = empirical_dist
                else:
                    global_dist = empirical_dist.mean(dim=0)
                
                running_dist = getattr(self, f'running_dist_{scale_name}')
                updated_dist = self.momentum * running_dist + (1 - self.momentum) * global_dist
                setattr(self, f'running_dist_{scale_name}', updated_dist)
        
        # 2. 计算多尺度分布一致性损失
        consistency_loss = 0
        if len(scale_distributions) > 1:
            distributions = list(scale_distributions.values())
            for i in range(len(distributions)):
                for j in range(i + 1, len(distributions)):
                    if distributions[i].dim() == distributions[j].dim():
                        consistency_loss += F.kl_div(
                            torch.log(distributions[i] + 1e-8),
                            distributions[j],
                            reduction='batchmean'
                        )
            consistency_loss *= self.consistency_weight
        
        # 3. 统计信息
        analysis_info = {
            'scale_distributions': scale_distributions,
            'scale_weights': scale_weights,
            'consistency_loss': consistency_loss,
            'total_pixels': targets.numel(),
            'high_building_ratio': (targets > 20.0).float().mean().item(),
            'mean_height': targets.mean().item(),
            'max_height': targets.max().item()
        }
        
        return pixel_weights, analysis_info


class AdaptiveMultiScaleLoss(nn.Module):
    """
    自适应多尺度损失函数
    
    结合分布感知重加权和Focal回归损失
    """
    def __init__(self,
                 base_loss_type: str = 'smooth_l1',
                 scale_weights: Optional[Dict[str, float]] = None,
                 focal_params: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.base_loss_type = base_loss_type
        self.scale_weights = scale_weights or {
            'scale_1': 0.125, 'scale_2': 0.25, 'scale_3': 0.5, 'scale_4': 1.0
        }
        
        # Focal loss参数
        focal_params = focal_params or {}
        self.focal_alpha = focal_params.get('alpha', 0.25)
        self.focal_gamma = focal_params.get('gamma', 2.0)
        self.height_threshold = focal_params.get('height_threshold', 20.0)
        
        # 基础损失函数
        if base_loss_type == 'smooth_l1':
            self.base_loss_fn = F.smooth_l1_loss
        elif base_loss_type == 'mse':
            self.base_loss_fn = F.mse_loss
        elif base_loss_type == 'mae':
            self.base_loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {base_loss_type}")
    
    def focal_regression_loss(self,
                            pred: torch.Tensor,
                            target: torch.Tensor,
                            weight_map: torch.Tensor) -> torch.Tensor:
        """
        Focal回归损失 - 专门关注高建筑物（困难样本）
        
        Args:
            pred: 预测值 [B, H, W]
            target: 真实值 [B, H, W]
            weight_map: 分布权重图 [B, H, W]
        """
        # 基础损失
        base_loss = self.base_loss_fn(pred, target, reduction='none')
        
        # Focal调制因子
        abs_error = torch.abs(pred - target)
        
        # 高度因子：高建筑物 = 困难样本
        height_factor = torch.sigmoid((target - self.height_threshold) / 5.0)
        
        # 错误调制因子：错误越大，权重越高
        error_mean = abs_error.mean() + 1e-8
        error_ratio = abs_error / error_mean
        error_ratio = torch.clamp(error_ratio, min=1e-8, max=10.0)  # 限制范围
        
        # 限制gamma防止pow爆炸
        gamma = min(self.focal_gamma, 2.0)  # 限制gamma最大为2
        error_factor = torch.pow(error_ratio, gamma)
        # error_factor = torch.pow(abs_error / error_mean, self.focal_gamma)
        
        # 综合调制
        focal_modulation = (
            self.focal_alpha * height_factor * error_factor + 
            (1 - self.focal_alpha)
        )
        
        # 应用分布权重和focal权重
        weighted_loss = base_loss * focal_modulation * weight_map
        
        return weighted_loss.mean()
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: torch.Tensor,
                pixel_weights: Dict[str, torch.Tensor],
                masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 计算加权多尺度损失
        
        Args:
            predictions: 多尺度预测 {'scale_1': [B,H,W], ...}
            targets: 真实标签 [B, H, W]
            pixel_weights: 像素权重 {'scale_1': [B,H,W], ...}
            masks: 可选掩码 [B, H, W]
        """
        total_loss = 0
        scale_losses = {}
        
        for scale_name, pred in predictions.items():
            if scale_name not in self.scale_weights:
                continue
            
            # 调整目标尺寸
            if targets.shape != pred.shape:
                target_resized = F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=pred.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)
            else:
                target_resized = targets
            
            # 获取权重图
            weight_map = pixel_weights.get(scale_name, torch.ones_like(target_resized))
            
            # 应用掩码
            if masks is not None:
                if masks.shape != pred.shape:
                    mask_resized = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=pred.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(1)
                else:
                    mask_resized = masks
                
                weight_map = weight_map * mask_resized
            
            # 计算focal损失
            scale_loss = self.focal_regression_loss(pred, target_resized, weight_map)
            
            # 应用尺度权重
            weighted_scale_loss = scale_loss * self.scale_weights[scale_name]
            
            scale_losses[f'{scale_name}_loss'] = scale_loss.item()
            total_loss += weighted_scale_loss
        
        return {
            'loss': total_loss,
            **scale_losses
        }


class DistributionMonitor(nn.Module):
    """
    分布监控器 - 用于训练过程中的可视化和调试
    """
    def __init__(self, num_height_bins: int = 50, max_height: float = 100.0):
        super().__init__()
        self.num_height_bins = num_height_bins
        self.max_height = max_height
        
        # 记录训练过程中的统计信息
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('cumulative_high_ratio', torch.tensor(0.0))
        self.register_buffer('cumulative_mean_height', torch.tensor(0.0))
    
    def update_statistics(self, targets: torch.Tensor):
        """更新累积统计信息"""
        with torch.no_grad():
            self.step_count += 1
            
            # 高建筑比例
            high_ratio = (targets > 20.0).float().mean()
            self.cumulative_high_ratio = (
                (self.cumulative_high_ratio * (self.step_count - 1) + high_ratio) / 
                self.step_count
            )
            
            # 平均高度
            mean_height = targets.mean()
            self.cumulative_mean_height = (
                (self.cumulative_mean_height * (self.step_count - 1) + mean_height) /
                self.step_count
            )
    
    def get_statistics(self) -> Dict[str, float]:
        """获取当前统计信息"""
        return {
            'steps': self.step_count.item(),
            'avg_high_building_ratio': self.cumulative_high_ratio.item(),
            'avg_mean_height': self.cumulative_mean_height.item()
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.step_count.fill_(0)
        self.cumulative_high_ratio.fill_(0.0)
        self.cumulative_mean_height.fill_(0.0)