import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
class ProjectionBlock(nn.Module):
    """Projection Block - 基于论文Figure 4右上"""
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.GELU()  # 论文明确使用GELU
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 从1D latent space映射到2D space的准备
        return self.projection(x)

class GradientAdaptiveEdgeModule(nn.Module):
    """梯度特性自适应边缘增强模块"""
    def __init__(self, channels: int):
        super().__init__()
        
        # 梯度提取器（使用Sobel算子的可学习版本）
        self.gradient_conv_x = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.gradient_conv_y = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        
        # 初始化为Sobel算子
        self._init_gradient_conv()
        
        # 梯度特征分析
        self.gradient_analyzer = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(channels, channels // 4, 1, 1, 0),
            nn.ReLU(True)
        )
        
        # 梯度一致性判别器
        self.consistency_detector = nn.Sequential(
            nn.Conv2d(channels // 4, channels // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // 4, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
        # 边缘增强卷积
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        
        # 残差权重（可学习的融合系数）- 初始化为更小的值
        self.alpha = nn.Parameter(torch.tensor(0.01))  # 从0.1改为0.01
        
        # 全局信息融合层
        self.global_fusion = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1, 1, 0),
            nn.ReLU(True)
        )
        
        # 添加权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    # def __init__(self, channels: int):
    #     super().__init__()
        
    #     # 梯度提取器（使用Sobel算子的可学习版本）
    #     self.gradient_conv_x = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
    #     self.gradient_conv_y = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        
    #     # 初始化为Sobel算子
    #     self._init_gradient_conv()
        
    #     # 梯度特征分析
    #     self.gradient_analyzer = nn.Sequential(
    #         nn.Conv2d(channels * 2, channels, 1, 1, 0),  # 融合x,y梯度
    #         nn.ReLU(True),
    #         nn.Conv2d(channels, channels // 4, 1, 1, 0),
    #         nn.ReLU(True)
    #     )
        
    #     # 梯度一致性判别器
    #     self.consistency_detector = nn.Sequential(
    #         nn.Conv2d(channels // 4, channels // 4, 3, 1, 1),
    #         nn.ReLU(True),
    #         nn.Conv2d(channels // 4, 1, 1, 1, 0),
    #         nn.Sigmoid()  # 输出0-1的权重，1表示建筑物边缘
    #     )
        
    #     # 边缘增强卷积
    #     self.edge_conv = nn.Sequential(
    #         nn.Conv2d(channels, channels, 3, 1, 1),
    #         nn.ReLU(True),
    #         nn.Conv2d(channels, channels, 3, 1, 1)
    #     )
        
    #     # 残差权重（可学习的融合系数）
    #     self.alpha = nn.Parameter(torch.tensor(0.1))
    #     #全局信息融合层
    #     self.global_fusion = nn.Sequential(
    #         nn.Conv2d(channels + 1, channels, 1, 1, 0),
    #         nn.ReLU(True)
    #     )
        
    def _init_gradient_conv(self):
        """初始化梯度卷积为Sobel算子"""
        # Sobel X
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(self.gradient_conv_x.out_channels, 1, 1, 1)
        self.gradient_conv_x.weight.data = sobel_x
        
        # Sobel Y  
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(self.gradient_conv_y.out_channels, 1, 1, 1)
        self.gradient_conv_y.weight.data = sobel_y
    def compute_gradient_consistency(self, grad_x: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
        """计算梯度一致性特征 - 修复形状问题"""
        B, C, H, W = grad_x.shape
        
        # 计算梯度幅值
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-4)
        
        # 计算梯度方向
        grad_direction = torch.atan2(grad_y + 1e-8, grad_x + 1e-8)
        
        # 裁剪梯度值防止极值
        grad_magnitude = torch.clamp(grad_magnitude, min=1e-6, max=100.0)
        grad_direction = torch.clamp(grad_direction, min=-np.pi, max=np.pi)
        
        # 【修复】简化一致性计算，避免复杂的unfold操作
        # 使用平均池化来计算局部统计信息
        kernel_size = 3
        padding = kernel_size // 2
        
        # 使用卷积来计算局部均值和方差
        avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=padding)
        
        # 计算局部方向均值
        local_mean_direction = avg_pool(grad_direction)
        
        # 计算局部方向方差
        direction_diff = grad_direction - local_mean_direction
        direction_var = avg_pool(direction_diff ** 2)
        
        # 计算局部幅值均值（用作权重）
        local_mean_magnitude = avg_pool(grad_magnitude)
        
        # 计算一致性：方差越小，一致性越高
        consistency = torch.exp(-direction_var)  # 使用指数函数，方差小时一致性高
        
        # 用幅值加权
        consistency = consistency * (local_mean_magnitude + 1e-6)
        
        # 归一化到[0,1]
        consistency = torch.sigmoid(consistency)
        
        # 确保输出形状正确
        assert consistency.shape == (B, C, H, W), f"Expected shape {(B, C, H, W)}, got {consistency.shape}"
        
        return consistency       
    # def compute_gradient_consistency(self, grad_x: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
    #     """计算梯度一致性特征 - 修复数值稳定性问题"""
    #     # 计算梯度幅值 - 增大epsilon防止sqrt(0)
    #     grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-4)  # 从1e-6增大到1e-4
        
    #     # 计算梯度方向 - 添加数值稳定性检查
    #     grad_direction = torch.atan2(grad_y + 1e-8, grad_x + 1e-8)  # 添加小量防止除零
        
    #     # 裁剪梯度值防止极值
    #     grad_magnitude = torch.clamp(grad_magnitude, min=1e-6, max=100.0)
    #     grad_direction = torch.clamp(grad_direction, min=-np.pi, max=np.pi)
        
    #     # 计算局部方向一致性（使用滑动窗口）
    #     kernel_size = 3  # 从5改为3，减少计算复杂度
    #     unfold = nn.Unfold(kernel_size, padding=kernel_size//2)
        
    #     # 展开局部窗口
    #     direction_patches = unfold(grad_direction)  # [B, k*k, H*W]
    #     magnitude_patches = unfold(grad_magnitude)  # [B, k*k, H*W]
        
    #     # 添加数值稳定性检查
    #     magnitude_sum = magnitude_patches.sum(dim=1) + 1e-4  # 增大epsilon
    #     magnitude_sum = torch.clamp(magnitude_sum, min=1e-4)  # 确保不会太小
        
    #     # 计算加权方向标准差（低标准差=高一致性）
    #     weighted_mean = (direction_patches * magnitude_patches).sum(dim=1) / magnitude_sum
        
    #     # 计算方差时添加数值稳定性
    #     diff_squared = (direction_patches - weighted_mean.unsqueeze(1)) ** 2
    #     weighted_variance = (diff_squared * magnitude_patches).sum(dim=1) / magnitude_sum
        
    #     # 裁剪方差防止负值或过大值
    #     weighted_variance = torch.clamp(weighted_variance, min=1e-6, max=10.0)
        
    #     weighted_std = torch.sqrt(weighted_variance)
        
    #     # 重塑回空间维度
    #     B, C, H, W = grad_x.shape
    #     consistency = 1.0 - torch.tanh(weighted_std)  # 高一致性=1，低一致性=0
    #     consistency = consistency.view(B, C, H, W)
        
    #     # 最终裁剪
    #     consistency = torch.clamp(consistency, min=0.0, max=1.0)
        
    #     return consistency
    # def compute_gradient_consistency(self, grad_x: torch.Tensor, grad_y: torch.Tensor) -> torch.Tensor:
    #     """计算梯度一致性特征"""
    #     B, C, H, W = grad_x.shape
        
    #     # 对所有通道计算平均梯度
    #     grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6).mean(dim=1, keepdim=True)  # [B, 1, H, W]
    #     grad_direction = torch.atan2(grad_y, grad_x).mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
    #     # 计算局部方向一致性（使用滑动窗口）
    #     kernel_size = 5
    #     unfold = nn.Unfold(kernel_size, padding=kernel_size//2)
        
    #     # 展开局部窗口
    #     direction_patches = unfold(grad_direction)  # [B, k*k, H*W]
    #     magnitude_patches = unfold(grad_magnitude)  # [B, k*k, H*W]
        
    #     # 计算加权方向标准差（低标准差=高一致性）
    #     weighted_mean = (direction_patches * magnitude_patches).sum(dim=1) / (magnitude_patches.sum(dim=1) + 1e-6)  # [B, H*W]
    #     weighted_std = torch.sqrt(
    #         ((direction_patches - weighted_mean.unsqueeze(1)) ** 2 * magnitude_patches).sum(dim=1) / 
    #         (magnitude_patches.sum(dim=1) + 1e-6)
    #     )  # [B, H*W]
        
    #     # 重塑回空间维度
    #     consistency = 1.0 - torch.tanh(weighted_std)  # [B, H*W]
    #     consistency = consistency.view(B, 1, H, W)  # [B, 1, H, W]
        
    #     # 扩展到所有通道
    #     consistency = consistency.expand(B, C, H, W)  # [B, C, H, W]
        
    #     return consistency        
    # def forward(self, x: torch.Tensor,global_map:torch.Tensor = None) -> torch.Tensor:
    #     if global_map is not None:
    #         #调整全局图大小来匹配特征
    #         global_map_resized = F.interpolate(
    #             global_map, 
    #             size=x.shape[-2:], 
    #             mode='bilinear', 
    #             align_corners=True)
    #         x_with_global = torch.cat([x, global_map_resized], dim=1)  # [B, C+1, H, W]
    #         x = self.global_fusion(x_with_global)  # [B, C, H, W]
    #     """
    #     Args:
    #         x: 输入特征 [B, C, H, W]
    #     Returns:
    #         增强后的特征 [B, C, H, W]
    #     """
    #     # 1. 提取梯度
    #     grad_x = self.gradient_conv_x(x)
    #     grad_y = self.gradient_conv_y(x)
        
    #     # 2. 分析梯度特征
    #     grad_features = torch.cat([grad_x, grad_y], dim=1)
    #     grad_analysis = self.gradient_analyzer(grad_features)
        
    #     # 3. 计算梯度一致性（判别建筑物边缘）
    #     consistency = self.compute_gradient_consistency(grad_x, grad_y)
        
    #     # 4. 生成自适应权重
    #     edge_weight = self.consistency_detector(grad_analysis)  # [B, 1, H, W]
    #     if global_map is not None:
    #         global_weight = F.interpolate(
    #             global_map, 
    #             size=edge_weight.shape[-2:], 
    #             mode='bilinear', 
    #             align_corners=True
    #         )
    #         edge_weight = edge_weight * (0.5+0.5*global_weight)  # 全局信息调制
        
    #     # 5. 结合梯度一致性
    #     edge_weight = edge_weight * consistency.mean(dim=1, keepdim=True)
        
    #     # 6. 边缘增强
    #     edge_enhanced = self.edge_conv(x)
        
    #     # 7. 自适应融合
    #     output = x + self.alpha * edge_weight * edge_enhanced
        
    #     return output
    def forward(self, x: torch.Tensor, global_map: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
            global_map: 全局梯度置信图 (可选)
        Returns:
            增强后的特征 [B, C, H, W]
        """
        # 输入检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN or Inf detected in input to GradientAdaptiveEdgeModule")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 处理全局信息融合
        if global_map is not None:
            # 调整全局图大小来匹配特征
            global_map_resized = F.interpolate(
                global_map, 
                size=x.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
            
            # 检查global_map中的异常值
            if torch.isnan(global_map_resized).any() or torch.isinf(global_map_resized).any():
                print("Warning: NaN or Inf detected in global_map")
                global_map_resized = torch.nan_to_num(global_map_resized, nan=0.5, posinf=1.0, neginf=0.0)
            
            x_with_global = torch.cat([x, global_map_resized], dim=1)  # [B, C+1, H, W]
            x = self.global_fusion(x_with_global)  # [B, C, H, W]
            
            # 检查融合后的结果
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Warning: NaN or Inf detected after global fusion")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 1. 提取梯度
        grad_x = self.gradient_conv_x(x)
        grad_y = self.gradient_conv_y(x)
        
        # 检查梯度计算结果
        if torch.isnan(grad_x).any() or torch.isinf(grad_x).any():
            print("Warning: NaN or Inf detected in grad_x")
            grad_x = torch.nan_to_num(grad_x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(grad_y).any() or torch.isinf(grad_y).any():
            print("Warning: NaN or Inf detected in grad_y")
            grad_y = torch.nan_to_num(grad_y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 2. 分析梯度特征
        grad_features = torch.cat([grad_x, grad_y], dim=1)
        grad_analysis = self.gradient_analyzer(grad_features)
        
        # 检查梯度分析结果
        if torch.isnan(grad_analysis).any() or torch.isinf(grad_analysis).any():
            print("Warning: NaN or Inf detected in grad_analysis")
            grad_analysis = torch.nan_to_num(grad_analysis, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 3. 计算梯度一致性（判别建筑物边缘）
        consistency = self.compute_gradient_consistency(grad_x, grad_y)
        
        # 检查一致性计算结果
        if torch.isnan(consistency).any() or torch.isinf(consistency).any():
            print("Warning: NaN or Inf detected in consistency")
            consistency = torch.nan_to_num(consistency, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 4. 生成自适应权重
        edge_weight = self.consistency_detector(grad_analysis)  # [B, 1, H, W]
        
        # 检查权重
        if torch.isnan(edge_weight).any() or torch.isinf(edge_weight).any():
            print("Warning: NaN or Inf detected in edge_weight")
            edge_weight = torch.nan_to_num(edge_weight, nan=0.5, posinf=1.0, neginf=0.0)
        
        if global_map is not None:
            global_weight = F.interpolate(
                global_map, 
                size=edge_weight.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
            edge_weight = edge_weight * (0.5 + 0.5 * global_weight)  # 全局信息调制
        
        # 5. 结合梯度一致性
        edge_weight = edge_weight * consistency.mean(dim=1, keepdim=True)
        
        # 6. 边缘增强
        edge_enhanced = self.edge_conv(x)
        
        # 检查边缘增强结果
        if torch.isnan(edge_enhanced).any() or torch.isinf(edge_enhanced).any():
            print("Warning: NaN or Inf detected in edge_enhanced")
            edge_enhanced = torch.nan_to_num(edge_enhanced, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 7. 自适应融合 - 降低alpha值
        alpha = torch.clamp(self.alpha, min=0.0, max=0.1)  # 限制alpha范围
        enhancement = alpha * edge_weight * edge_enhanced
        
        # 最终检查
        if torch.isnan(enhancement).any() or torch.isinf(enhancement).any():
            print("Warning: NaN or Inf detected in enhancement")
            enhancement = torch.nan_to_num(enhancement, nan=0.0, posinf=1.0, neginf=-1.0)
        
        output = x + enhancement
        
        # 输出检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: NaN or Inf detected in output")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return output
class RefineBlock(nn.Module):
    """Refine Block - 基于论文Figure 4右下"""
    def __init__(self, in_channels: int,deeper_channels:int =None):
        super().__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(True)
        )
        if deeper_channels is not None and deeper_channels != in_channels:
            self.channel_proj=nn.Conv2d(deeper_channels,in_channels,1,1,0)
        else:
            self.channel_proj=None
        # 修改1: 添加梯度自适应边缘增强模块
        self.edge_enhance = GradientAdaptiveEdgeModule(in_channels)
        #添加通道对齐层
        # self.channel_align = None
    #     self._channel_align_layers=nn.ModuleDict()
    # def _get_channel_align_layer(self, from_channels:int, to_channels:int):
    #     key=f"{from_channels}_to_{to_channels}"
    #     if key not in self._channel_align_layers:
    #         self._channel_align_layers[key]=nn.Conv2d(from_channels, to_channels,1,1,0)
    #     return self._channel_align_layers[key]

    def forward(self, fn: torch.Tensor, fn_plus_1: torch.Tensor = None,
                global_map:torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            fn: 当前层特征
            fn_plus_1: 更深层特征 (可选)
            global_map: 全局梯度置信图 (可选)
        """
        
        if fn_plus_1 is not None:
            # print('RefineBlock前向传播')
            # print(fn.shape[1],fn_plus_1.shape[1],fn.shape[-2:],fn_plus_1.shape[-2:])
            if self.channel_proj is not None:
                fn_plus_1=self.channel_proj(fn_plus_1)
            combined = fn + fn_plus_1
            output = self.conv_relu(combined)
            # return self.conv_relu(combined)
        else:
            output = self.conv_relu(fn)
        output = self.edge_enhance(output,global_map)  # 添加边缘增强
        return output
    
            # return self.conv_relu(fn)

class ResolutionAgnosticDecoder(nn.Module):
    """Resolution-Agnostic Decoder - 基于论文Figure 4"""
    def __init__(self, 
                 embed_dim: int, 
                 num_register_tokens: int = 0):
        super().__init__()
        self.num_register_tokens = num_register_tokens
        
        # 论文Figure 4: 4个不同尺度的处理通道
        self.out_channels = [256, 512, 1024, 1024]  # 论文中的配置
        
        # Projection Blocks for each scale
        self.projection_blocks = nn.ModuleList([
            ProjectionBlock(embed_dim, out_channel) 
            for out_channel in self.out_channels
        ])
        
        # Resize operations (论文Figure 4左侧)
        self.resize_ops = nn.ModuleList([
            nn.ConvTranspose2d(self.out_channels[0], self.out_channels[0], 4, 4, 0),  # 4x upsample
            nn.ConvTranspose2d(self.out_channels[1], self.out_channels[1], 2, 2, 0),  # 2x upsample  
            nn.Identity(),  # No change
            nn.Conv2d(self.out_channels[3], self.out_channels[3], 3, 2, 1)  # 0.5x downsample
        ])
        
        # Refine Blocks for multi-scale fusion
        # self.refine_blocks = nn.ModuleList([
        #     RefineBlock(out_channel) for out_channel in self.out_channels
        # ])
        self.refine_blocks = nn.ModuleList([
            RefineBlock(self.out_channels[0]),
            RefineBlock(self.out_channels[1],self.out_channels[0]),
            RefineBlock(self.out_channels[2],self.out_channels[1]),
            RefineBlock(self.out_channels[3],self.out_channels[2]),
        ])

        # Final prediction heads for each scale
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel//2, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channel//2, 32, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, 1, 1, 0),
                nn.ReLU(True)
            ) for out_channel in self.out_channels
        ])
        
    def forward(self, 
               scale_features: List[torch.Tensor], 
               patch_h: int, 
               patch_w: int,
               global_gradient_map: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            scale_features: 来自Scale Modulator的4个尺度特征
            patch_h, patch_w: patch的高度和宽度
        Returns:
            dict: 包含4个尺度预测结果的字典
        """
        processed_features = []
        
        # 1. Projection: 1D to 2D conversion
        for i, (feat, proj_block) in enumerate(zip(scale_features, self.projection_blocks)):
            # 移除cls token和register tokens，只保留patch tokens
            if self.num_register_tokens > 0:
                patch_tokens = feat[:, 1 + self.num_register_tokens:, :]
            else:
                patch_tokens = feat[:, 1:, :]  # 移除cls token
            
            # Projection
            projected = proj_block(patch_tokens)  # [B, N, out_channel]
            
            # Reshape to 2D: [B, N, C] -> [B, C, H, W]
            B = projected.shape[0]
            projected_2d = projected.permute(0, 2, 1).reshape(
                B, self.out_channels[i], patch_h, patch_w
            )
            
            # Resize operations
            resized = self.resize_ops[i](projected_2d)
            processed_features.append(resized)
        
        # 2. Multi-scale refinement (论文的核心融合策略)
        refined_features = []
        for i in range(len(processed_features)):
            if i == 0:
                # 最深层特征，无需融合
                refined = self.refine_blocks[i](
                    processed_features[i],
                    global_map=global_gradient_map)
            else:
                # 与更深层特征融合
                # 需要调整尺寸匹配
                deeper_feat = F.interpolate(
                    refined_features[i-1], 
                    size=processed_features[i].shape[-2:],
                    mode='bilinear', 
                    align_corners=True
                )
                refined = self.refine_blocks[i](processed_features[i], deeper_feat,global_map=global_gradient_map)
            
            refined_features.append(refined)
        
        # 3. Generate multi-scale predictions
        predictions = {}
        for i, (feat, pred_head) in enumerate(zip(refined_features, self.prediction_heads)):
            height_map = pred_head(feat)  # [B, 1, H, W]
            
            # 论文描述：upsample to match input image resolution before calculating loss
            target_size = (patch_h * 14, patch_w * 14)  # 恢复到原始尺寸
            height_map = F.interpolate(
                height_map, 
                size=target_size,
                mode='bilinear', 
                align_corners=True
            )
            
            predictions[f'scale_{i+1}'] = height_map.squeeze(1)
            
        return predictions