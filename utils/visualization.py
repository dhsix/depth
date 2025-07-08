import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from typing import Optional, Tuple, List, Dict
from pathlib import Path

class DepthVisualizer:
    """深度图可视化工具"""
    
    @staticmethod
    def colorize_depth(depth: np.ndarray, 
                      vmin: Optional[float] = None, 
                      vmax: Optional[float] = None,
                      cmap: str = 'plasma') -> np.ndarray:
        """将深度图转换为彩色图像"""
        
        # 处理无效值
        valid_mask = np.isfinite(depth)
        if not np.any(valid_mask):
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        
        depth_clean = depth.copy()
        
        # 设置值域
        if vmin is None:
            vmin = np.min(depth_clean[valid_mask])
        if vmax is None:
            vmax = np.max(depth_clean[valid_mask])
        
        # 归一化
        if vmax > vmin:
            depth_norm = (depth_clean - vmin) / (vmax - vmin)
        else:
            depth_norm = np.zeros_like(depth_clean)
        
        depth_norm = np.clip(depth_norm, 0, 1)
        
        # 应用colormap
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap)
        colored = colormap(depth_norm)
        
        # 转换为uint8
        colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # 处理无效区域（设为黑色）
        colored_uint8[~valid_mask] = 0
        
        return colored_uint8
    
    @staticmethod
    def create_comparison_grid(images: List[np.ndarray], 
                             titles: List[str],
                             ncols: int = 3,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """创建对比网格图"""
        
        nrows = (len(images) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (img, title) in enumerate(zip(images, titles)):
            row = i // ncols
            col = i % ncols
            
            if img.ndim == 3 and img.shape[2] == 3:
                axes[row, col].imshow(img)
            else:
                axes[row, col].imshow(img, cmap='plasma')
            
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(images), nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_prediction_batch(images: torch.Tensor,
                                 predictions: torch.Tensor,
                                 targets: torch.Tensor,
                                 save_path: Optional[str] = None,
                                 max_samples: int = 4) -> plt.Figure:
        """可视化一个batch的预测结果"""
        
        batch_size = min(images.shape[0], max_samples)
        
        # 转换为numpy
        if images.is_cuda:
            images = images.cpu()
        if predictions.is_cuda:
            predictions = predictions.cpu()
        if targets.is_cuda:
            targets = targets.cpu()
        
        images_np = images.numpy()
        predictions_np = predictions.numpy()
        targets_np = targets.numpy()
        
        # 反归一化图像
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        vis_images = []
        vis_titles = []
        
        for i in range(batch_size):
            # 原图像
            img = images_np[i].transpose(1, 2, 0)
            img = img * std + mean
            img = np.clip(img, 0, 1)
            vis_images.append(img)
            vis_titles.append(f'Input {i+1}')
            
            # 真实深度
            target_colored = DepthVisualizer.colorize_depth(targets_np[i])
            vis_images.append(target_colored / 255.0)
            vis_titles.append(f'GT {i+1}')
            
            # 预测深度
            pred_colored = DepthVisualizer.colorize_depth(
                predictions_np[i], 
                vmin=np.min(targets_np[i]), 
                vmax=np.max(targets_np[i])
            )
            vis_images.append(pred_colored / 255.0)
            vis_titles.append(f'Pred {i+1}')
        
        # 创建网格图
        fig = DepthVisualizer.create_comparison_grid(
            vis_images, vis_titles, ncols=3,
            figsize=(12, 4 * batch_size)
        )
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_training_curves(metrics_history: Dict[str, List],
                           save_path: Optional[str] = None) -> plt.Figure:
        """绘制训练曲线"""
        
        # 找出所有损失相关的指标
        loss_metrics = [k for k in metrics_history.keys() if 'loss' in k.lower()]
        other_metrics = [k for k in metrics_history.keys() if 'loss' not in k.lower() and k != 'epoch']
        
        n_plots = len(loss_metrics) + (1 if other_metrics else 0)
        
        if n_plots == 0:
            return None
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        epochs = metrics_history.get('epoch', range(len(list(metrics_history.values())[0])))
        
        # 绘制损失曲线
        for i, metric in enumerate(loss_metrics):
            if metric in metrics_history:
                axes[i].plot(epochs, metrics_history[metric], label=metric)
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].set_title(f'{metric} vs Epoch')
                axes[i].grid(True)
                axes[i].legend()
        
        # 绘制其他指标
        if other_metrics and len(axes) > len(loss_metrics):
            ax = axes[len(loss_metrics)]
            for metric in other_metrics:
                if metric in metrics_history:
                    ax.plot(epochs, metrics_history[metric], label=metric)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metrics')
            ax.set_title('Other Metrics vs Epoch')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig

def save_prediction_samples(model, data_loader, save_dir: str, num_samples: int = 10):
    """保存预测样本"""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            
            images = batch['image']
            targets = batch['depth']
            
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()
            
            # 预测
            predictions = model.predict(images)
            if isinstance(predictions, dict):
                predictions = predictions.get('scale_4', list(predictions.values())[-1])
            
            # 可视化
            fig = DepthVisualizer.visualize_prediction_batch(
                images, predictions, targets, max_samples=2
            )
            
            # 保存
            fig.savefig(save_path / f'sample_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Saved {min(num_samples, len(data_loader))} prediction samples to {save_path}")