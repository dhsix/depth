import torch
from typing import Dict, Any, Iterator

def create_optimizer(model_parameters: Iterator[torch.nn.Parameter], 
                    config: Dict[str, Any]) -> torch.optim.Optimizer:
    """创建优化器"""
    
    optimizer_type = config.get('optimizer', 'adamw').lower()
    learning_rate = config.get('learning_rate', 5e-6)
    weight_decay = config.get('weight_decay', 0.01)
    
    if optimizer_type == 'adamw':
        return torch.optim.AdamW(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_type == 'adam':
        return torch.optim.Adam(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            model_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=config.get('momentum', 0.9),
            nesterov=config.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

def create_scheduler(optimizer: torch.optim.Optimizer, 
                    config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """创建学习率调度器"""
    
    scheduler_type = config.get('scheduler', 'constant').lower()
    
    if scheduler_type == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 50),
            eta_min=config.get('min_lr', 1e-8)
        )
    
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('step_size', 20),
            gamma=config.get('gamma', 0.5)
        )
    
    elif scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get('milestones', [30, 45]),
            gamma=config.get('gamma', 0.1)
        )
    
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('gamma', 0.95)
        )
    
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('patience', 5),
            verbose=True
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")