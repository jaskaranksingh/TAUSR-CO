import torch.nn as nn
import torch

class ECMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        normed = self.norm(x)
        contrast = x - normed
        return x + contrast * self.channel_attn(normed)
