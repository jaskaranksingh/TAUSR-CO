# utils/metrics.py

import torch
import torch.nn.functional as F

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value

def ssim(pred, target, window_size=11, size_average=True):
    # Simplified fast SSIM (channel-wise mean)
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    sigma_pred = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu_pred ** 2
    sigma_target = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu_target ** 2
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred * mu_target

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map
