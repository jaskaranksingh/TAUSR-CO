import torch
import torch.nn as nn
from models.fsfe import FSFEBlock
from models.snr import SNRBlock
from models.da import DABlock
from models.ecm import ECMBlock
from models.epm import EPMBlock
from models.fft_attention import FFTAttentionBlock

class TAUSR_CO(nn.Module):
    def __init__(self, in_channels=1, mid_channels=64):
        super(TAUSR_CO, self).__init__()
        
        # Shallow layer
        self.fsfe = FSFEBlock(in_channels, mid_channels)
        
        # Structural layer
        self.snr = SNRBlock(mid_channels)
        self.da = DABlock(mid_channels)
        self.fft_struct = FFTAttentionBlock(mid_channels)

        # Contextual layer
        self.tem = nn.Identity()  # Placeholder for full TEM if separate
        self.ecm = ECMBlock(mid_channels)
        self.epm = EPMBlock(mid_channels)
        self.fft_context = FFTAttentionBlock(mid_channels)

        # Skip and fusion
        self.fuse_conv = nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.output_layer = nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Shallow parsing
        f_shallow = self.fsfe(x)  # (B, C, H, W)

        # Structural parsing + intra-layer attention
        f_struct = self.snr(f_shallow)
        f_struct = self.da(f_struct)
        f_struct = self.fft_struct(f_struct)

        # Contextual parsing + intra-layer contrast
        f_context = self.tem(f_shallow)
        f_context = self.ecm(f_context)
        f_context = self.epm(f_context)
        f_context = self.fft_context(f_context)

        # Inter-layer fusion with skip from shallow
        fused = torch.cat([f_shallow, f_struct, f_context], dim=1)
        fused = self.fuse_conv(fused)

        # Upsampling
        upsampled = self.pixel_shuffle(fused)
        out = self.output_layer(upsampled)
        return out