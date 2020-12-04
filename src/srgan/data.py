from torchvision import transforms
import torch
import torch.nn as nn

from PIL import Image
import os
class SRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, hr_size, scale_factor=2):
        self.hr_size = hr_size
        self.dataset_path = dataset_path
        self.frames = os.listdir(dataset_path)
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(hr_size),
            transforms.ToTensor(),
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_size[0]//scale_factor,hr_size[1]//scale_factor)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = Image.open(os.path.join(self.dataset_path, self.frames[idx]))
        return self.lr_transform(frame), self.hr_transform(frame)
