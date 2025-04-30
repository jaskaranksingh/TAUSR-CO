import torch
import torch.nn as nn

class FusionBlock(nn.Module):
    def __init__(self, channels, upscale_factor=2):
        super().__init__()
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.output = nn.Conv2d(channels // (upscale_factor ** 2), 1, kernel_size=3, padding=1)

    def forward(self, f_shallow, f_struct, f_context):
        x = torch.cat([f_shallow, f_struct, f_context], dim=1)
        x = self.fuse(x)
        x = self.pixel_shuffle(x)
        return self.output(x)
