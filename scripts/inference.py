#!/usr/bin/env python3
"""
推理脚本
用法:
    python scripts/inference.py --model_path checkpoints/best_model.pth --input image.jpg
    python scripts/inference.py --config configs/depth2elevation.yaml --input_dir ./test_images
"""

import argparse
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

from models import create_model
from utils.visualization import DepthVisualizer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Run inference on depth estimation models')
    
    # 模型相关
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    
    # 输入输出
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--input_dir', type=str, help='Input directory with images')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='Output directory')
    
    # 推理参数
    parser.add_argument('--input_size', type=int, default=448, help='Input size for model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (for directory input)')
    
    # 可视化选项
    parser.add_argument('--save_raw', action='store_true', help='Save raw depth values')
    parser.add_argument('--save_colored', action='store_true', default=True, help='Save colored depth map')
    parser.add_argument('--colormap', type=str, default='plasma', help='Colormap for visualization')
    
    return parser.parse_args()

def load_model(model_path, config_path=None, device='cuda'):
    """加载模型"""
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 获取配置
    if config_path:
        from scripts.train import load_config_from_file
        config = load_config_from_file(config_path)
    elif 'config' in checkpoint:
        config_dict = checkpoint['config']
        if config_dict.get('model_name') == 'depth2elevation':
            from configs.depth2elevation_config import Depth2ElevationConfig
            config = Depth2ElevationConfig(**config_dict)
        else:
            from configs.base_config import BaseConfig
            config = BaseConfig(**config_dict)
    else:
        # 使用默认配置
        from configs.depth2elevation_config import Depth2ElevationConfig
        config = Depth2ElevationConfig()
    
    # 创建模型
    model = create_model(config.to_dict())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model

def process_single_image(model, image_path, output_dir, args):
    """处理单张图像"""
    
    # 加载图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    print(f"Processing: {image_path}")
    
    # 记录推理时间
    start_time = time.time()
    
    # 模型推理
    with torch.no_grad():
        depth_map = model.infer_image(image, input_size=args.input_size)
    
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.3f}s")
    
    # 准备输出路径
    image_name = Path(image_path).stem
    output_path = Path(output_dir) / image_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存原图
    cv2.imwrite(str(output_path / "input.jpg"), image)
    
    # 保存原始深度值
    if args.save_raw:
        np.save(output_path / "depth_raw.npy", depth_map)
        
        # 也保存为16位TIFF（保持精度）
        depth_uint16 = (depth_map * 1000).astype(np.uint16)  # 转换为毫米
        cv2.imwrite(str(output_path / "depth_raw.tif"), depth_uint16)
    
    # 保存彩色深度图
    if args.save_colored:
        colored_depth = DepthVisualizer.colorize_depth(depth_map, cmap=args.colormap)
        cv2.imwrite(str(output_path / "depth_colored.jpg"), colored_depth)
    
    # 创建对比图
    # 调整图像尺寸以匹配深度图
    h, w = depth_map.shape
    image_resized = cv2.resize(image, (w, h))
    
    # 创建对比图
    comparison = np.hstack([image_resized, colored_depth])
    cv2.imwrite(str(output_path / "comparison.jpg"), comparison)
    
    print(f"Results saved to: {output_path}")
    return True

def process_image_directory(model, input_dir, output_dir, args):
    """处理图像目录"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    # 查找图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No image files found in: {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # 处理每张图像
    success_count = 0
    for image_file in image_files:
        try:
            if process_single_image(model, image_file, output_dir, args):
                success_count += 1
        except Exception as e:
            print(f"Failed to process {image_file}: {e}")
    
    print(f"Successfully processed {success_count}/{len(image_files)} images")

def main():
    """主函数"""
    args = parse_args()
    
    # 检查输入参数
    if not args.input and not args.input_dir:
        print("Error: Either --input or --input_dir must be provided")
        sys.exit(1)
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载模型
        print("Loading model...")
        model = load_model(args.model_path, args.config, args.device)
        
        # 处理输入
        if args.input:
            # 单张图像
            process_single_image(model, args.input, output_dir, args)
        else:
            # 图像目录
            process_image_directory(model, args.input_dir, output_dir, args)
        
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Inference failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()