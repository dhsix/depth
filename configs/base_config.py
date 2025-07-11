import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class BaseConfig:
    """基础配置类"""
    
    # 项目设置
    project_name: str = "depth_estimation"
    experiment_name: str = "baseline"
    seed: int = 42
    device: str = "cuda"
    
    # 数据设置
    data_root: str = "/data/remote_sensing_datasets"  # 存储服务器上的数据根目录
    dataset_name: str = "GAMUS"  # GAMUS, DFC2019, Vaihingen
    input_size: int = 448
    patch_size: int = 14
    batch_size: int = 16
    num_workers: int = 8
    trainer_type:str ="depth"
    # 训练设置
    num_epochs: int = 50
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "constant"
    
    # 验证和保存
    val_interval: int = 5
    save_interval: int = 10
    max_keep_ckpts: int = 3
    
    # 路径设置
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    result_dir: str = "./results"
    
    # 模型设置 (子类需要重写)
    model_name: str = "base_model"
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # 损失函数设置
    loss_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后处理，创建必要的目录"""
        # 创建项目相关目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 检查数据集路径(但不创建，因为数据在存储服务器上)
        dataset_path = self.get_dataset_path()
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path {dataset_path} not accessible!")
            print(f"Please ensure:")
            print(f"  1. Storage server is mounted")
            print(f"  2. Data path is correctly configured: {self.data_root}")
            print(f"  3. Dataset exists at: {dataset_path}")
            print(f"Expected structure:")
            print(f"  {dataset_path}/")
            print(f"    train/images/, train/depths/")
            print(f"    val/images/, val/depths/")
            print(f"    test/images/, test/depths/")
        
    def get_dataset_path(self) -> str:
        """获取当前数据集的完整路径"""
        return os.path.join(self.data_root, self.dataset_name)
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从YAML配置文件加载配置"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
    def get_experiment_dir(self) -> str:
        """获取实验目录"""
        return os.path.join(self.result_dir, f"{self.dataset_name}_{self.model_name}_{self.experiment_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)