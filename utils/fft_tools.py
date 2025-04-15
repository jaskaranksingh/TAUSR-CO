import torch

def fft2d(x):
    return torch.fft.fft2(x)

def ifft2d(x):
    return torch.fft.ifft2(x)