import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import Generator
from utils import Config, PSNR, SSIM

import os
import random
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from PIL import Image
import numpy as np
import cv2

def image_inference(model, image_file, device):
    model.to(device)
    model.eval()
    scale = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(Config['img_size']),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                        std=[4.367,4.464,4.444]),
        torchvision.transforms.ToPILImage(),
        # torchvision.transforms.Scale(Config['img_size'])
    ])


    with torch.set_grad_enabled(False):
        img = torchvision.transforms.functional.to_tensor(Image.open(image_file))
        img = scale(img).unsqueeze(0)
        _,out = model(img)
    return transform(out)[0]

def replace_background(image, bg):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(Config['lower_green']), np.array(Config['upper_green']))
    mask = cv2.bitwise_not(mask)
    mask = cv2.medianBlur(mask,7)
    image[mask==0]=[0,0,0]
    bg[mask!=0]=[0,0,0]
    return bg+image

def video_inference(model, video_file, old_bg_file, bg_file, device):
    model.to(device)
    model.eval()
    scale = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(Config['img_size']),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                        std=[4.367,4.464,4.444]),
        torchvision.transforms.ToPILImage(),
        # torchvision.transforms.Scale(Config['img_size'])
    ])
    resize_cv = (Config['img_size'][1]*Config['scale'], Config['img_size'][0]*Config['scale'])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (640,480))
    with torch.set_grad_enabled(False):
        cap = cv2.VideoCapture(video_file)
        bg = scale(torchvision.transforms.functional.to_tensor(Image.open(bg_file)))
        old_bg = cv2.imread(old_bg_file)
        counter=0
        while cap.isOpened():
            ret, frame = cap.read()
            counter+=1
            print('Frame: '+str(counter), end='\r')
            # if not ret or frame is not None:
            #     break
            if frame is None: break
            frame = cv2.resize(frame, resize_cv)
            old_bg = cv2.resize(old_bg, resize_cv)
            frame = replace_background(frame, old_bg)
            cv2.imwrite('frame.png', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.asarray(Image.fromarray(frame))
            frame = scale(frame)
            inputs = torch.cat([frame, bg],0).unsqueeze(0)
            _,out=model(inputs)
            out = np.asarray(transform(out[0].cpu()))[:, :, ::-1].copy()
            # print(out.shape)
            # cv2.imshow('result',out)
            writer.write(out)
            # cv2.waitKey()
    writer.close()

if __name__=='__main__':
    model = Generator()
    model.load_state_dict(torch.load('generator_80.pth',map_location='cpu')())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    video_file = '/home/adityan/Documents/datasets/videos/old/Blonde_Woman_Upset_4.mp4'
    bg_file = '/home/adityan/Documents/datasets/backgrounds/zoom-4.png'
    old_bg_file = '/home/adityan/Documents/datasets/backgrounds/zoom-1v2.png'

    video_inference(model, video_file, old_bg_file, bg_file, device)
