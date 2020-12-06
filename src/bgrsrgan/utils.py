Config={}

Config['debug']=True
Config['use_cuda']=True
Config['train_set_path']='/home/ubuntu/DIV2K/'
# Config['test_set_path']='/home/adityan/EE599_Deep_Learning_Project/src/data/images'
Config['checkpoint_path']='/home/ubuntu/srgan_new/'

Config['scale']=2
Config['n_colors']=3
Config['n_resblocks']=6
Config['n_feats']=64
Config['epochs']=100

Config['batch_size']=8
Config['num_workers']=4
Config['img_size']=(240,320) #(480,640)

Config['generator_lr']=1e-4
Config['discriminator_lr']=1e-4
Config['optimizer']='Adam'
Config['skip_threshold']=1e8
Config['tensorboard_log']=True

Config['generator_checkpoint']=None
Config['discriminator_checkpoint']=None

Config['lower_green'] = [50,80,80]
Config['upper_green'] = [90,255,255]

import torch
import numpy as np
import cv2

class PSNR:
    def __init__(self):
        self.name = 'PSNR'

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1-img2)**2)
        return 20 * torch.log10(255.0/torch.sqrt(mse))

class SSIM:
    def __init__(self):
        self.name = 'SSIM'

    @staticmethod
    def __call__(img1, img2):
        ssims=[]
        for i in range(3):
            ssims.append(SSIM()._ssim(img1[i].cpu().detach().numpy(), img2[i].cpu().detach().numpy()))
        return np.array(ssims).mean()

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1*img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq+sigma2_sq+C2))

        return ssim_map.mean()
