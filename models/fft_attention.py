# models/fft_attention.py

import torch
import torch.nn as nn

class FFTAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(FFTAttentionBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        freq = torch.fft.fft2(x)
        magnitude = torch.abs(freq)
        phase = torch.angle(freq)

        attn = self.sigmoid(self.conv1x1(magnitude.real))
        magnitude_attended = magnitude * attn

        real = magnitude_attended * torch.cos(phase)
        imag = magnitude_attended * torch.sin(phase)
        freq_attended = torch.complex(real, imag)

        x_out = torch.fft.ifft2(freq_attended).real

        return x_out
