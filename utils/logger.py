import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def setup_logger(name: str, 
                log_file: Optional[str] = None, 
                level: int = logging.INFO,
                console_output: bool = True) -> logging.Logger:
    """设置日志记录器"""
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # 创建实验目录
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.experiment_dir / f"training_{timestamp}.log"
        
        self.logger = setup_logger(
            name=f"training_{experiment_name}",
            log_file=str(log_file),
            level=logging.INFO
        )
        
        # 记录训练指标
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """记录每个epoch的结果"""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        
        # 记录额外指标
        for key, value in kwargs.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # 日志输出
        log_msg = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
        for key, value in kwargs.items():
            log_msg += f", {key}={value:.4f}"
        
        self.logger.info(log_msg)
    
    def log_info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def log_config(self, config):
        """记录配置信息"""
        self.logger.info("=" * 50)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("=" * 50)
        
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config
            
        for key, value in config_dict.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("=" * 50)
    
    def save_metrics(self):
        """保存训练指标"""
        import json
        metrics_file = self.experiment_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)