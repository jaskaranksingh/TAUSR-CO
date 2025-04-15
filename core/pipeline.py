import torch
from models.fsfe import FSFEBlock
from models.snr import SNRBlock
from models.da import DABlock
from models.ecm import ECMBlock
from models.epm import EPMBlock
from models.fft_attention import FFTAttentionBlock

def run_pipeline(config_path=None):
    print(f"Using config: {config_path}")
    
    dummy_input = torch.randn(1, 1, 128, 128)
    fsfe = FSFEBlock(1, 64)
    snr = SNRBlock(64)
    da = DABlock(64)
    ecm = ECMBlock(64)
    epm = EPMBlock(64)
    fft = FFTAttentionBlock(64)

    x = fsfe(dummy_input)
    x = snr(x)
    x = da(x)
    x = fft(x)
    x = ecm(x)
    x = epm(x)

    print("Forward pass success. Output shape:", x.shape)