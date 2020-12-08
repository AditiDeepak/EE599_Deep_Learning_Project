import torch
import torchvision

import cv2
import numpy as np
from PIL import Image
import random
import os

from utils import Config

class BGRSRDataset(torch.utils.data.Dataset):
    def __init__(self, frames_path, backgrounds_path):
        self.frames = [os.path.join(frames_path, file) for file in os.listdir(frames_path) if file.endswith('.png') or file.endswith('.jpg')]
        self.backgrounds = [os.path.join(backgrounds_path, file) for file in os.listdir(backgrounds_path) if file.endswith('.png') or file.endswith('.jpg') or file.endswith('jpeg')]

        self.img_size = Config['img_size']
        self.scale_factor = Config['scale']

        self.resize_cv = (self.img_size[1]*self.scale_factor, self.img_size[0]*self.scale_factor)
        self.transform = torchvision.transforms.Compose([#torchvision.transforms.RandomCrop((Config['img_size'][0]*Config['scale'],Config['img_size'][1]*Config['scale'])),
                                    torchvision.transforms.ToTensor()
                                    ])
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std = [0.229, 0.224, 0.225])

        self.scale = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                                torchvision.transforms.Resize(Config['img_size']),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std = [0.229, 0.224, 0.225])
                                                                ])

    def __len__(self):
        return len(self.frames)

    def replace_background(self, image, bg):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(Config['lower_green']), np.array(Config['upper_green']))
        mask = cv2.bitwise_not(mask)
        mask = cv2.medianBlur(mask,7)
        image[mask==0]=[0,0,0]
        bg[mask!=0]=[0,0,0]
        return bg+image

    def __getitem__(self, idx):
        img = cv2.imread(self.frames[idx])
        orig_bg, new_bg = random.sample(self.backgrounds, k=2)
        orig_bg = cv2.imread(orig_bg)
        new_bg = cv2.imread(new_bg)
        # cv2.imwrite('new_bg.jpg', new_bg)
        img = cv2.resize(img, self.resize_cv)
        orig_bg = cv2.resize(orig_bg, self.resize_cv)
        new_bg = cv2.resize(new_bg, self.resize_cv)
        orig_img = self.replace_background(img.copy(), orig_bg.copy())
        new_img = self.replace_background(img.copy(), new_bg.copy())

        # print(Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)))
        orig_img = self.transform(Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)))
        new_img = self.transform(Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)))
        new_bg = self.transform(Image.fromarray(cv2.cvtColor(new_bg, cv2.COLOR_BGR2RGB)))
        # from torchvision.utils import save_image
        # save_image(orig_img, 'orig_img.png')
        # save_image(new_bg, 'new_bg.png')
        # save_image(new_img, 'new_img.png')
        return self.scale(orig_img), self.scale(new_bg), self.scale(new_img), self.normalize(new_img)
