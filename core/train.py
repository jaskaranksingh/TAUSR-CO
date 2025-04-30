# core/train.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import psnr, ssim
from utils.losses import CombinedLoss
from core.scheduler import build_scheduler

def train(model, train_dataset, val_dataset, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = build_scheduler(optimizer, mode=config['scheduler'], num_epochs=config['epochs'])
    loss_fn = CombinedLoss()

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_psnr = 0

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = loss_fn(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        psnr_val, ssim_val = validate(model, val_loader, device)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(psnr_val)
            else:
                scheduler.step()

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            torch.save(model.state_dict(), config['save_path'])

        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, PSNR {psnr_val:.2f}, SSIM {ssim_val:.4f}")

def validate(model, val_loader, device):
    model.eval()
    psnr_total, ssim_total = 0, 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            psnr_total += psnr(sr, hr).item()
            ssim_total += ssim(sr, hr).item()

    return psnr_total / len(val_loader), ssim_total / len(val_loader)
