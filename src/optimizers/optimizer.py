import torch
import torch.nn as nn
from torch.optim import optimizer

def build_optimizer(model, config):
    optimizer_type = config['optimizer']
    lr = config['true_lr']

    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), 
                                lr=lr, 
                                weight_decay=config['adam_decay'])
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(),
                                lr=lr,
                                weight_decay=config['adamw_decay'])
    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), 
                                lr=lr, 
                                weight_decay=config['sgd_decay'],
                                momentum=config['sgd_momentum'])
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {optimizer_type} is not a valid optimizer!")
