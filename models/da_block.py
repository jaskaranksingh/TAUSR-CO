import torch
import torch.nn as nn

class DABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conv(x)
        x_flat = x.view(b, c, -1).permute(2, 0, 1)
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        x_attn = x_attn.permute(1, 2, 0).view(b, c, h, w)
        return x_attn
