import logging
import os
import sys
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

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
    
    # 文件输出 - 启用文件记录
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class TrainingLogger:
    """训练日志记录器 - 增强版"""
    
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
        
        # 记录batch级别的指标
        self.batch_metrics_history = []
        
        # 配置信息存储
        self.config_info = None
        
    def log_config(self, config):
        """记录配置信息 - 增强版本，更好地处理yaml配置"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 EXPERIMENT CONFIGURATION")
        self.logger.info("=" * 80)
        
        # 处理不同类型的配置
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = vars(config) if hasattr(config, '__dict__') else str(config)
        
        # 存储配置信息
        self.config_info = config_dict
        
        # 分类显示配置信息
        self._log_config_section("📊 Data Configuration", config_dict, [
            'data_root', 'dataset_name', 'input_size', 'patch_size', 
            'batch_size', 'num_workers'
        ])
        
        self._log_config_section("🧠 Model Configuration", config_dict, [
            'model_name', 'model_config', 'encoder', 'embed_dim', 'num_heads'
        ])
        
        self._log_config_section("🎯 Training Configuration", config_dict, [
            'num_epochs', 'learning_rate', 'weight_decay', 'optimizer', 
            'scheduler', 'val_interval', 'save_interval'
        ])
        
        self._log_config_section("📉 Loss Configuration", config_dict, [
            'loss_config', 'trainer_type', 'use_multi_scale_output'
        ])
        
        # 如果有分布重加权配置
        if 'reweight_config' in config_dict and config_dict['reweight_config'].get('enable', False):
            self._log_config_section("⚖️ Distribution Reweighting Configuration", 
                                    config_dict, ['reweight_config'])
        
        self._log_config_section("🔧 System Configuration", config_dict, [
            'device', 'seed', 'experiment_name', 'checkpoint_dir', 'log_dir'
        ])
        
        # 记录其他配置
        other_keys = set(config_dict.keys()) - {
            'data_root', 'dataset_name', 'input_size', 'patch_size', 'batch_size', 
            'num_workers', 'model_name', 'model_config', 'num_epochs', 'learning_rate', 
            'weight_decay', 'optimizer', 'scheduler', 'val_interval', 'save_interval',
            'loss_config', 'trainer_type', 'use_multi_scale_output', 'reweight_config',
            'device', 'seed', 'experiment_name', 'checkpoint_dir', 'log_dir'
        }
        
        if other_keys:
            self._log_config_section("📝 Other Configuration", config_dict, list(other_keys))
        
        self.logger.info("=" * 80)
        
        # 保存配置到文件
        self._save_config_to_file(config_dict)
        
    def _log_config_section(self, section_title: str, config_dict: Dict[str, Any], keys: list):
        """记录配置的某个部分"""
        self.logger.info(f"\n{section_title}:")
        self.logger.info("-" * 40)
        
        for key in keys:
            if key in config_dict:
                value = config_dict[key]
                if isinstance(value, dict):
                    self.logger.info(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        self.logger.info(f"    {sub_key}: {sub_value}")
                else:
                    self.logger.info(f"  {key}: {value}")
    
    def _save_config_to_file(self, config_dict: Dict[str, Any]):
        """保存配置到文件"""
        # 保存为JSON格式
        config_json_file = self.experiment_dir / "experiment_config.json"
        with open(config_json_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        
        # 如果可能，也保存为YAML格式
        try:
            config_yaml_file = self.experiment_dir / "experiment_config.yaml"
            with open(config_yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.logger.warning(f"Failed to save config as YAML: {e}")
        
        self.logger.info(f"💾 Configuration saved to: {config_json_file}")
    
    def log_batch_progress(self, epoch: int, batch_idx: int, total_batches: int, 
                          metrics: Dict[str, float], lr: float = None):
        """记录batch级别的训练进度"""
        # 每100个batch或最后几个batch记录一次
        if batch_idx % 100 == 0 or batch_idx >= total_batches - 5:
            # 计算进度百分比
            progress = (batch_idx + 1) / total_batches * 100
            
            # 格式化指标
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            
            # 学习率信息
            lr_str = f", LR: {lr:.2e}" if lr is not None else ""
            
            # 进度条
            bar_length = 30
            filled_length = int(bar_length * (batch_idx + 1) // total_batches)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            log_message = (
                f"📈 Epoch {epoch:3d} | "
                f"[{bar}] {progress:6.2f}% | "
                f"Batch {batch_idx + 1:4d}/{total_batches:4d} | "
                f"{metrics_str}{lr_str}"
            )
            
            self.logger.info(log_message)
        
        # 记录batch级别的指标历史（采样记录，避免文件过大）
        if batch_idx % 100 == 0:
            batch_record = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'metrics': metrics.copy(),
                'lr': lr,
                'timestamp': datetime.now().isoformat()
            }
            self.batch_metrics_history.append(batch_record)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        """记录每个epoch的结果 - 深度估计专用增强版本"""
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        
        # 记录额外指标
        for key, value in kwargs.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # 格式化输出
        self.logger.info("=" * 80)
        self.logger.info(f"🏁 EPOCH {epoch:3d} COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Train Loss: {train_loss:.6f}")
        if not (isinstance(val_loss, float) and (val_loss != val_loss)):  # 检查不是NaN
            self.logger.info(f"📊 Val Loss:   {val_loss:.6f}")
        
        # 专门处理深度估计指标
        if kwargs:
            self._log_depth_metrics(kwargs)
        
        self.logger.info("=" * 80)
    
    def _log_depth_metrics(self, metrics: Dict[str, Any]):
        """专门记录深度估计指标"""
        # 分类整理指标
        loss_metrics = {}
        depth_metrics = {}
        accuracy_metrics = {}
        other_metrics = {}
        
        for key, value in metrics.items():
            if 'loss' in key.lower():
                loss_metrics[key] = value
            elif key in ['MAE', 'RMSE', 'SI_RMSE', 'AbsRel', 'SqRel']:
                depth_metrics[key] = value
            elif key.startswith('δ') or 'accuracy' in key.lower():
                accuracy_metrics[key] = value
            else:
                other_metrics[key] = value
        
        # 显示损失相关指标
        if loss_metrics:
            self.logger.info("💸 Loss Breakdown:")
            for key, value in loss_metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
                    self.logger.info(f"   {key}: {value:.6f}")
                else:
                    self.logger.info(f"   {key}: {value}")
        
        # 显示深度估计核心指标
        if depth_metrics:
            self.logger.info("📏 Depth Estimation Metrics:")
            # 按重要性排序显示
            metric_order = ['MAE', 'RMSE', 'SI_RMSE', 'AbsRel', 'SqRel']
            for metric in metric_order:
                if metric in depth_metrics:
                    value = depth_metrics[metric]
                    if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
                        # 根据指标类型选择合适的精度
                        if metric in ['MAE', 'RMSE', 'SI_RMSE']:
                            self.logger.info(f"   📐 {metric}: {value:.4f}")
                        else:  # AbsRel, SqRel
                            self.logger.info(f"   📊 {metric}: {value:.4f}")
            
            # 显示剩余的深度指标
            for key, value in depth_metrics.items():
                if key not in metric_order:
                    if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
                        self.logger.info(f"   📊 {key}: {value:.4f}")
        
        # 显示准确率指标
        if accuracy_metrics:
            self.logger.info("🎯 Accuracy Metrics:")
            for key, value in accuracy_metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
                    # 准确率通常是0-1之间，显示为百分比
                    if 0 <= value <= 1:
                        self.logger.info(f"   ✅ {key}: {value:.1%}")
                    else:
                        self.logger.info(f"   ✅ {key}: {value:.4f}")
        
        # 显示其他指标
        if other_metrics:
            self.logger.info("📈 Other Metrics:")
            for key, value in other_metrics.items():
                if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
                    self.logger.info(f"   📊 {key}: {value:.4f}")
                else:
                    self.logger.info(f"   📊 {key}: {value}")
    
    def log_depth_summary(self, metrics: Dict[str, float]):
        """记录深度估计指标摘要"""
        self.logger.info("=" * 80)
        self.logger.info("📊 DEPTH ESTIMATION PERFORMANCE SUMMARY")
        self.logger.info("=" * 80)
        
        # 核心指标摘要
        core_metrics = ['MAE', 'RMSE', 'SI_RMSE']
        self.logger.info("🎯 Core Performance Metrics:")
        for metric in core_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, (int, float)):
                    self.logger.info(f"   📐 {metric}: {value:.4f}")
        
        # 相对误差指标
        rel_metrics = ['AbsRel', 'SqRel']
        if any(m in metrics for m in rel_metrics):
            self.logger.info("📊 Relative Error Metrics:")
            for metric in rel_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, (int, float)):
                        self.logger.info(f"   📊 {metric}: {value:.4f}")
        
        # 阈值准确率
        threshold_metrics = [k for k in metrics.keys() if k.startswith('δ')]
        if threshold_metrics:
            self.logger.info("🎯 Threshold Accuracy:")
            for metric in sorted(threshold_metrics):
                value = metrics[metric]
                if isinstance(value, (int, float)):
                    self.logger.info(f"   ✅ {metric}: {value:.1%}")
        
        self.logger.info("=" * 80)
    
    def log_info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告"""
        self.logger.warning(f"⚠️  {message}")
    
    def log_error(self, message: str):
        """记录错误"""
        self.logger.error(f"❌ {message}")
    
    def log_success(self, message: str):
        """记录成功信息"""
        self.logger.info(f"✅ {message}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """记录模型信息"""
        self.logger.info("=" * 80)
        self.logger.info("🏗️  MODEL INFORMATION")
        self.logger.info("=" * 80)
        
        for key, value in model_info.items():
            if key == 'total_parameters' or key == 'trainable_parameters':
                # 格式化参数数量
                if isinstance(value, int):
                    value_str = f"{value:,} ({value/1e6:.2f}M)" if value > 1e6 else f"{value:,}"
                    self.logger.info(f"  {key}: {value_str}")
                else:
                    self.logger.info(f"  {key}: {value}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 80)
    
    def log_training_start(self, total_epochs: int, train_batches: int, val_batches: int):
        """记录训练开始信息"""
        self.logger.info("🚀" * 20)
        self.logger.info("🚀 TRAINING STARTED")
        self.logger.info("🚀" * 20)
        self.logger.info(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"🎯 Total Epochs: {total_epochs}")
        self.logger.info(f"📊 Train Batches: {train_batches}")
        self.logger.info(f"📊 Val Batches: {val_batches}")
        self.logger.info("🚀" * 20)
    
    def log_training_complete(self, total_time: float, best_val_loss: float = None):
        """记录训练完成信息"""
        self.logger.info("🎉" * 20)
        self.logger.info("🎉 TRAINING COMPLETED")
        self.logger.info("🎉" * 20)
        self.logger.info(f"📅 End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"⏱️  Total Time: {total_time/3600:.2f} hours")
        if best_val_loss is not None:
            self.logger.info(f"🏆 Best Val Loss: {best_val_loss:.6f}")
        self.logger.info("🎉" * 20)
    
    def log_dataset_info(self, train_size: int, val_size: int, test_size: int = None):
        """记录数据集信息"""
        self.logger.info("=" * 80)
        self.logger.info("📂 DATASET INFORMATION")
        self.logger.info("=" * 80)
        self.logger.info(f"  📊 Train Samples: {train_size:,}")
        self.logger.info(f"  📊 Val Samples: {val_size:,}")
        if test_size is not None:
            self.logger.info(f"  📊 Test Samples: {test_size:,}")
        self.logger.info(f"  📊 Total Samples: {train_size + val_size + (test_size or 0):,}")
        self.logger.info("=" * 80)
    
    def log_checkpoint_saved(self, epoch: int, path: str, is_best: bool = False):
        """记录检查点保存"""
        checkpoint_type = "🏆 BEST" if is_best else "💾"
        self.logger.info(f"{checkpoint_type} Checkpoint saved at epoch {epoch}: {path}")
    
    def save_metrics(self):
        """保存训练指标"""
        # 保存epoch级别的指标
        metrics_file = self.experiment_dir / "training_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        
        # 保存batch级别的指标
        if self.batch_metrics_history:
            batch_metrics_file = self.experiment_dir / "batch_metrics.json"
            with open(batch_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.batch_metrics_history, f, indent=2, ensure_ascii=False, default=str)
        
        self.log_success(f"Metrics saved to {self.experiment_dir}")
    
    def get_summary(self) -> str:
        """获取训练摘要"""
        if not self.metrics_history['train_loss']:
            return "No training data available"
        
        best_epoch = self.metrics_history['val_loss'].index(min(self.metrics_history['val_loss']))
        best_train_loss = self.metrics_history['train_loss'][best_epoch]
        best_val_loss = self.metrics_history['val_loss'][best_epoch]
        
        summary = f"""
        📋 TRAINING SUMMARY
        ==================
        🎯 Total Epochs: {len(self.metrics_history['train_loss'])}
        🏆 Best Epoch: {best_epoch + 1}
        📉 Best Train Loss: {best_train_loss:.6f}
        📉 Best Val Loss: {best_val_loss:.6f}
        """
        return summary