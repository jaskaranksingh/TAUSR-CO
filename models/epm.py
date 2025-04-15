import torch.nn as nn

class EPMBlock(nn.Module):
    def __init__(self, channels):
        super(EPMBlock, self).__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.norm(x)