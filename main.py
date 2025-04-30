# main.py

import yaml
import torch
from torch.utils.data import DataLoader
from models.tausr_model import TAUSRCO
from core.dataset import UltrasoundDataset
from core.train import train
from core.infer import infer

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('configs/config.yaml')

    model = TAUSRCO(
        in_channels=config['model']['in_channels'],
        mid_channels=config['model']['mid_channels']
    )

    if config['mode'] == 'train':
        train_dataset = UltrasoundDataset(config['dataset']['root'], mode='train')
        val_dataset = UltrasoundDataset(config['dataset']['root'], mode='val')
        train(model, train_dataset, val_dataset, config['train'])

    elif config['mode'] == 'infer':
        test_dataset = UltrasoundDataset(config['dataset']['root'], mode='val')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        infer(model, test_loader, config['infer'])

    else:
        raise ValueError("Invalid mode. Use 'train' or 'infer'.")

if __name__ == "__main__":
    main()
