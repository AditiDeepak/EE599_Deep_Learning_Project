from model import Generator
from utils import Config

import os
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torchvision import transforms, PSNR, SSIM
from torch.utils.tensorboard import SummaryWriter

def inference(video_path, model, result_path, device):
    '''
    video_path location to low resolution video
    '''
    model.eval()
    model.to(device)

    cap = cv2.VideoCapture(video_path)

    input_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    output_transform = transforms.Compose([
        transforms.Normalize(mean=[-2.118,-2.036,-1.804],
                            std=[4.367, 4.464, 4.444]),
        transforms.ToPILImage(),
        transforms.Scale(Config['img_size'])
    ])
    while cap.isOpened():
        ret, frame = cap.read()
        # convert frame to tensor and transform
