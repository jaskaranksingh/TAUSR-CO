import torch.nn as nn

class ECMBlock(nn.Module):
    def __init__(self, channels):
        super(ECMBlock, self).__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        normed = self.norm(x)
        contrast = x - normed
        attn = self.channel_attn(normed)
        return x + contrast * attn