# import torch
import torch.nn as nn

from models.fsfe_block import FSFEBlock
from models.transformer_block import TextureEnhancementTransformer
from models.fft_attention import FFTAttentionBlock
from models.snr_block import SNRBlock
from models.da_block import DABlock
from models.ecm_block import ECMBlock
from models.epm_block import EPMBlock
from models.fusion_block import FusionBlock

class TAUSRCO(nn.Module):
    """
    - FSFE (Shallow): Structural edge and base feature extraction
    - Structural: Denoising + depth correction + frequency attention
    - Contextual: Transformer + contrast + edge + frequency attention
    """

    def __init__(self, in_channels=1, mid_channels=64, upscale=2):
        super().__init__()

        self.shallow_layer = FSFEBlock(in_channels, mid_channels)

        self.structural_layer = nn.Sequential(
            SNRBlock(mid_channels),
            DABlock(mid_channels),
            FFTAttentionBlock(mid_channels)
        )

        self.contextual_layer = nn.Sequential(
            TextureEnhancementTransformer(mid_channels),
            ECMBlock(mid_channels),
            EPMBlock(mid_channels),
            FFTAttentionBlock(mid_channels)
        )

        self.fusion = FusionBlock(mid_channels, upscale_factor=upscale)

    def forward(self, x):
        f_shallow = self.shallow_layer(x)
        f_structural = self.structural_layer(f_shallow) + f_shallow
        f_contextual = self.contextual_layer(f_shallow) + f_shallow
        return self.fusion(f_shallow, f_structural, f_contextual)
