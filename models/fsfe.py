import torch.nn as nn
import torchvision.models as models

class FSFEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FSFEBlock, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.resnet(x)
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.final_conv(x)
