# core/scheduler.py

import torch.optim.lr_scheduler as lr_scheduler

def build_scheduler(optimizer, mode='cosine', num_epochs=100):
    if mode == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif mode == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = None
    return scheduler
