# core/dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


#  Dataset must be in this form

# root/
#  └── train/
#      ├── LR/
#      ├── HR/
#  └── val/
#      ├── LR/
#      ├── HR/



class UltrasoundDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        super(UltrasoundDataset, self).__init__()
        self.root_dir = os.path.join(root_dir, mode)
        self.lr_dir = os.path.join(self.root_dir, 'LR')
        self.hr_dir = os.path.join(self.root_dir, 'HR')
        self.lr_images = sorted(os.listdir(self.lr_dir))
        self.hr_images = sorted(os.listdir(self.hr_dir))
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr = Image.open(lr_path).convert('L')
        hr = Image.open(hr_path).convert('L')
        lr = self.transform(lr)
        hr = self.transform(hr)
        return lr, hr
