import torch.nn as nn
import torch

class FFTAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(FFTAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        freq = torch.fft.fft2(x)
        amp = torch.abs(freq)
        attn = self.conv(amp.real)
        attn = self.sigmoid(attn)
        return x * attn