# 深度估计：遥感图像多尺度高度估计增强方法

本项目基于TGRS 2025论文"Depth2Elevation: Scale Modulation With Depth Anything Model for Single-View Remote Sensing Image Height Estimation"实现了增强版的Depth2Elevation模型，用于单视图遥感图像高度估计任务。

## 🚀 主要特性

- **基础模型集成**：基于Depth Anything Model (DAM)的遥感高度估计实现
- **多尺度架构**：尺度调制器和分辨率无关解码器处理不同空间尺度
- **边缘增强处理**：梯度自适应边缘增强模块，提升建筑物边界检测精度
- **分布感知训练**：标签分布平滑(LDS)技术解决高度分布不平衡问题
- **灵活配置**：支持多种模型变体和训练配置

## 🏗️ 架构概述

### 核心组件

1. **尺度调制器**：从最后4个transformer块提取多尺度特征
2. **分辨率无关解码器**：处理多尺度特征并输出4个不同尺度的高度图
3. **多尺度损失**：结合MSE、尺度不变和梯度损失的鲁棒训练

### 创新增强

1. **梯度自适应边缘模块**：使用可学习梯度算子增强建筑物边缘特征
2. **分布重加权机制**：通过自适应损失加权解决长尾高度分布问题

## 📁 项目结构

```
DEPTH2ELEVATION_PROJECT/
├── checkpoints/                      # 模型检查点和保存权重
├── configs/                          # 配置文件
│   ├── baseline_configs/             # 基线模型配置
│   ├── base_config.py                # 基础配置类
│   ├── depth2elevation_config.py     # 模型特定配置
│   ├── depth2elevation_gamus.yaml    # GAMUS数据集配置
│   ├── depth2elevation_GoogleHeight.yaml # GoogleHeight数据集配置
│   ├── depth2elevation_gra_gamus.yaml # 带梯度增强的配置
│   ├── depth2elevation_multiscale_gamus.yaml # 完整增强模型配置
│   ├── im2height_gamus.yaml          # IM2HEIGHT基线配置
│   └── imele_gamus.yaml              # IMELE基线配置
├── data/                             # 数据处理和加载
│   ├── __init__.py                   # 包初始化
│   ├── data_loader.py                # 数据加载器创建和管理
│   ├── datasets.py                   # 数据集实现
│   └── transforms.py                 # 数据增强和预处理
├── experiments/                      # 实验输出和结果
├── models/                           # 模型实现
│   ├── depth2elevation/              # 原始Depth2Elevation模型
│   │   ├── dinov2_layers/            # DINOv2相关层
│   │   ├── __init__.py
│   │   ├── decoder.py                # 解码器实现
│   │   ├── model.py                  # 主模型
│   │   └── scale_modulator.py        # 尺度调制器
│   ├── Gra_MultiScaleHeight/         # 增强模型（包含创新点）
│   │   ├── dinov2_layers/            # DINOv2相关层
│   │   ├── __init__.py
│   │   ├── decoder.py                # 带梯度增强的解码器
│   │   ├── distribution_reweighting.py # 分布重加权模块
│   │   ├── model.py                  # 增强主模型
│   │   └── scale_modulator.py        # 增强尺度调制器
│   ├── losses/                       # 损失函数实现
│   │   ├── __init__.py
│   │   ├── base_losses.py            # 基础损失函数
│   │   └── multi_scale_loss.py       # 多尺度损失
│   ├── __init__.py                   # 模型包初始化
│   └── base_model.py                 # 抽象基础模型类
├── scripts/                          # 执行脚本
│   ├── evaluate.py                   # 评估脚本
│   ├── inference.py                  # 推理脚本
│   └── train.py                      # 训练脚本
├── trainer/                          # 训练框架
│   ├── __init__.py                   # 包初始化
│   ├── base_trainer.py               # 抽象训练器类
│   └── depth_trainer.py              # 深度估计专用训练器
├── utils/                            # 工具函数
│   ├── __init__.py                   # 包初始化
│   ├── checkpoint.py                 # 模型检查点管理
│   ├── logger.py                     # 训练日志记录和监控
│   ├── metrics.py                    # 评估指标
│   ├── optimizers.py                 # 优化器和调度器创建
│   └── visualization.py              # 结果可视化
├── requirements.txt                  # Python依赖列表
└── README.md                         # 本文件
```

## 🗂️ 支持的数据集

1. **GAMUS**：几何感知多模态分割数据集，包含高分辨率卫星图像
2. **DFC2019**：IEEE GRSS数据融合竞赛2019数据集
3. **Vaihingen**：ISPRS基准数据集

### 数据结构

每个数据集应按以下方式组织：
```
dataset_root/
├── train/
│   ├── images/           # RGB图像
│   └── depths/           # 高度/深度图 (或labels/)
├── val/
│   ├── images/
│   └── depths/
└── test/ (可选)
    ├── images/
    └── depths/
```

## 🚀 快速开始

### 训练

```bash
# 使用YAML配置文件训练
python scripts/train.py --config configs/depth2elevation_multiscale_gamus.yaml

# 使用命令行参数训练
python scripts/train.py --model depth2elevation_multiscale --dataset GAMUS --epochs 50 --batch_size 4
```

### 评估

```bash
# 评估训练好的模型
python scripts/evaluate.py --model_path checkpoints/best_model.pth --dataset GAMUS

# 使用特定配置评估
python scripts/evaluate.py --config configs/depth2elevation_multiscale_gamus.yaml --checkpoint best
```

### 推理

```bash
# 单张图像推理
python scripts/inference.py --model_path checkpoints/best_model.pth --input image.jpg

# 批量推理
python scripts/inference.py --model_path checkpoints/best_model.pth --input_dir ./test_images
```

## ⚙️ 配置说明

### 模型变体

1. **depth2elevation**：论文原始模型
2. **depth2elevation_gra**：带梯度自适应边缘增强的模型
3. **depth2elevation_multiscale**：包含分布重加权的增强模型
4. **im2height**：IM2HEIGHT基线模型
5. **imele**：IMELE基线模型

### 关键配置参数

```yaml
# 模型配置
model_name: "depth2elevation_multiscale"
model_config:
  encoder: "vitb"                    # vits, vitb, vitl, vitg
  img_size: 448
  patch_size: 14
  pretrained_path: "path/to/dam_weights.pth"

# 训练配置
num_epochs: 50
learning_rate: 5.0e-6
batch_size: 4
val_interval: 5

# 增强功能
reweight_config:
  enable: true                       # 启用分布重加权
  num_height_bins: 50
  max_height: 100.0
  alpha: 0.7
```

## 📊 评估指标

- **MAE**：平均绝对误差
- **RMSE**：均方根误差
- **SI_RMSE**：尺度不变均方根误差
- **AbsRel**：绝对相对误差
- **SqRel**：平方相对误差
- **δ<1.25**：阈值精度指标

## 🤝 贡献

1. Fork仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📜 引用

如果您在研究中使用了此代码，请引用：

```bibtex
@article{hong2025depth2elevation,
  title={Depth2Elevation: Scale Modulation With Depth Anything Model for Single-View Remote Sensing Image Height Estimation},
  author={Hong, Zhongcheng and Wu, Tong and Xu, Zhiyuan and Zhao, Wufan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={63},
  pages={4504914},
  year={2025},
  publisher={IEEE}
}
```

## 📄 许可证

本项目采用MIT许可证。

## 🙏 致谢

- 原始Depth2Elevation论文和实现
- Depth Anything Model (DAM)基础模型
- IEEE GRSS提供的基准数据集