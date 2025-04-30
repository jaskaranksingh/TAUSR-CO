# utils/losses.py

import torch
import torch.nn as nn
from utils.metrics import ssim

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        return 1 - ssim(pred, target)

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        f_pred = self.feature_extractor(pred)
        f_target = self.feature_extractor(target)
        return nn.functional.l1_loss(f_pred, f_target)

class CombinedLoss(nn.Module):
    def __init__(self, ssim_weight=0.1, perc_weight=0.01):
        super(CombinedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perc_weight = perc_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred, target, perceptual=None):
        loss = self.l1(pred, target)
        loss += self.ssim_weight * self.ssim_loss(pred, target)
        if perceptual is not None:
            loss += self.perc_weight * perceptual(pred, target)
        return loss
