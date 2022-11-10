
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyWithLogits, self).__init__()

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

def construct_loss(config):
    loss_type = config['loss']
    if loss_type == 'Triplet':
        return nn.TripletMarginLoss(margin=config['triplet_margin'],
                                    p=config['triplet_p'],
                                    reduction=config['triplet_reduction'])
    elif loss_type == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'BCE':
        return BinaryCrossEntropyWithLogits()
    elif loss_type == 'ListWise':
        raise NotImplementedError()
    else:
        raise ValueError(f"TRAINER.LOSS = {loss_type} is not a valid loss function type!")
