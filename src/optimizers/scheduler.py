from torch.optim.lr_scheduler import MultiStepLR, \
    CosineAnnealingLR, ExponentialLR, LambdaLR

def build_scheduler(optimizer, config):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler_type = config['scheduler']
    scheduler = {'interval': config['scheduler_interval'],
                 'frequency': config['scheduler_frequency']}

    if scheduler_type == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, milestones=config['mslr_milestones'], gamma=config['mslr_gamma'])})
    elif scheduler_type == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, T_max=config['cosa_tmax'])})
    elif scheduler_type == 'LambdaLR':
        lambda1 = lambda epoch: epoch // 30
        lambda2 = lambda epoch: 0.9991 ** epoch
        scheduler.update(
            {'scheduler': LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])})
    elif scheduler_type == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, gamma=config['elr_gamma'])})
    else:
        raise NotImplementedError()
    print('scheduler: ', scheduler)
    return scheduler
