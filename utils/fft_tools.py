# utils/fft_tools.py


#Not used: For Future

import torch

def fft2d(x):
    return torch.fft.fft2(x)

def ifft2d(x):
    return torch.fft.ifft2(x)

def fft_magnitude(x):
    freq = fft2d(x)
    return torch.abs(freq)

def fft_phase(x):
    freq = fft2d(x)
    return torch.angle(freq)

def fft_shift(x):
    return torch.fft.fftshift(x)

def ifft_shift(x):
    return torch.fft.ifftshift(x)

def ifft_reconstruct(magnitude, phase):
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    complex_val = torch.complex(real, imag)
    return torch.fft.ifft2(complex_val)
