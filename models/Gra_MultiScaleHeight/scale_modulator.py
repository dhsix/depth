import torch
import torch.nn as nn
from functools import partial
from typing import List
import torch.nn.functional as F
class ScaleAdapter(nn.Module):
    """Scale Adapter - 基于论文Figure 3(c)的精确实现"""
    def __init__(self, embed_dim: int):
        super().__init__()
        # 基于论文：Down-projection maps to 128 dimensions
        self.down_projection = nn.Linear(embed_dim, 128)
        self.activation = nn.ReLU(True)  # 论文明确使用ReLU
        self.up_projection = nn.Linear(128, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 论文描述：residual connection with input feature
        identity = x
        x = self.down_projection(x)
        x = self.activation(x)
        x = self.up_projection(x)
        return x + identity  # Residual connection

class HeightBlock(nn.Module):
    """Height Block - 基于DINOv2 Transformer Block的修改版本"""
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 ffn_bias: bool = True,
                 drop_path: float = 0.0):
        super().__init__()
        
        # 导入DINOv2的组件（假设已经实现）
        try:
            from .dinov2_layers import MemEffAttention, Mlp
            print('Import dinov2_layer')
        except ImportError:
            # 如果没有DINOv2，使用标准实现
            from torch.nn import MultiheadAttention
            MemEffAttention = MultiheadAttention
            
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.norm1 = norm_layer(embed_dim)
        self.attn = MemEffAttention(
            embed_dim, 
            num_heads=num_heads,
            batch_first=True
        ) if hasattr(MemEffAttention, 'embed_dim') else nn.MultiheadAttention(
            embed_dim, 
            num_heads,
            batch_first=True
        )
        
        self.norm2 = norm_layer(embed_dim)
        
        # 标准MLP
        if 'Mlp' in locals():
            self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                nn.Dropout(0.1)
            )
        
        # 论文关键创新：集成trainable MLP after attention residual connection
        self.additional_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(0.1)
        )
        
        # Drop path (Stochastic Depth)
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard transformer with additional MLP
        if hasattr(self.attn, 'forward'):
            attn_out = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        else:
            attn_out = x + self.drop_path(self.attn(self.norm1(x)))
            
        attn_out = attn_out + self.additional_mlp(attn_out)  # 论文创新点
        x = attn_out + self.drop_path(self.mlp(self.norm2(attn_out)))
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
class GlobalGradientAnalyzer(nn.Module):
    """全局梯度分析模块 - 分析多尺度特征生成全局梯度置信图"""
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # 多尺度特征融合
        self.scale_fusion = nn.ModuleList([
            nn.Linear(embed_dim, 256) for _ in range(4)
        ])
        
        # 梯度模式分析网络
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128)
        )
        
        # 生成空间置信图
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 2x上采样
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 2x上采样
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()  # 输出0-1的置信度
        )
        
    def forward(self, scale_features: List[torch.Tensor], 
                patch_h: int, patch_w: int) -> torch.Tensor:
        """
        分析多尺度特征，生成全局梯度置信图
        
        Args:
            scale_features: 4个尺度的特征 [B, N, C]
            patch_h, patch_w: patch网格大小
        Returns:
            global_map: 全局梯度置信图 [B, 1, H, W]
        """
        B = scale_features[0].shape[0]
        
        # 1. 多尺度特征降维和融合
        compressed_features = []
        for feat, fusion_layer in zip(scale_features, self.scale_fusion):
            # 取全局平均（简化处理，也可以用更复杂的池化）
            if feat.shape[1] > 1:  # 如果有多个token
                global_feat = feat[:, 1:, :].mean(dim=1)  # 忽略cls token
            else:
                global_feat = feat.squeeze(1)
            
            compressed = fusion_layer(global_feat)
            compressed_features.append(compressed)
        
        # 2. 拼接多尺度特征
        multi_scale_feat = torch.cat(compressed_features, dim=1)  # [B, 256*4]
        
        # 3. 分析梯度模式
        pattern_feat = self.pattern_analyzer(multi_scale_feat)  # [B, 128]
        
        # 4. 重塑为空间特征图
        # 初始空间大小设为patch grid的1/4
        init_h, init_w = patch_h // 4, patch_w // 4
        spatial_feat = pattern_feat.view(B, 128, 1, 1)
        spatial_feat = spatial_feat.expand(B, 128, init_h, init_w)
        
        # 5. 生成全分辨率置信图
        confidence_map = self.spatial_decoder(spatial_feat)
        
        # 6. 上采样到原始分辨率
        target_size = (patch_h * 14, patch_w * 14)  # 匹配原始图像大小
        confidence_map = F.interpolate(
            confidence_map,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )
        
        return confidence_map
    
class ScaleModulator(nn.Module):
    """Scale Modulator - 论文核心创新，基于Figure 3(a)"""
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        
        # 论文明确：12个Height Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 12)]  # stochastic depth decay rule
        
        self.height_blocks = nn.ModuleList([
            HeightBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i]
            ) for i in range(12)
        ])
        
        # 论文明确：4个Scale Adapters，插入在最后4个blocks
        self.scale_adapters = nn.ModuleList([
            ScaleAdapter(embed_dim) for _ in range(4)
        ])
        
        # 论文Figure 3(a)显示：从最后4个blocks提取特征
        self.adapter_positions = [8, 9, 10, 11]  # 最后4个位置
        self.global_gradient_analyzer = GlobalGradientAnalyzer(embed_dim)  # 假设有一个全局梯度分析器
        
    def forward(self, x: torch.Tensor,patch_h:int =None, patch_w:int =None) -> List[torch.Tensor]:
        """
        Args:
            x: 来自patch embedding的特征 [B, N, C]
        Returns:
            list: 4个scale features [fs1, fs2, fs3, fs4]
        """
        """
        Args:
            x: 来自patch embedding的特征 [B, N, C]
            patch_h, patch_w: patch的高度和宽度（用于生成空间梯度图）
        Returns:
            tuple: (scale_features, global_gradient_map)
                - scale_features: 4个scale features [fs1, fs2, fs3, fs4]
                - global_gradient_map: 全局梯度置信图
        """
        scale_features = []
        
        # 通过12个Height Blocks
        for i, height_block in enumerate(self.height_blocks):
            x = height_block(x)
            
            # 在最后4个blocks处提取并处理特征
            if i in self.adapter_positions:
                adapter_idx = self.adapter_positions.index(i)
                adapted_feature = self.scale_adapters[adapter_idx](x)
                scale_features.append(adapted_feature)
        # 生成全局梯度置信图
        global_gradient_map = None
        if patch_h is not None and patch_w is not None:
            global_gradient_map = self.global_gradient_analyzer(scale_features, patch_h, patch_w)

        return scale_features,global_gradient_map 
