import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from model import Generator, Discriminator, get_feat_extractor
from model_edsr import content_loss, adversarial_loss
from utils import Config
from data import SRDataset

def train():
    args=Config
    generator = Generator(args)
    discriminator = Discriminator(args)
    feat_extractor = get_feat_extractor()
    dataset = SRDataset(dataset_path=args['train_set_path'],hr_size=args['hr_size'], scale_factor=args['scale'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=Config['batch_size'],
                    shuffle=True)
    test_dataset = SRDataset(args['test_set_path'],args['hr_size'], args['scale'])
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                    shuffle=True)

    if Config['optimizer']=='Adam':
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr = Config['lr'])
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr = Config['lr'])
    else:
        gen_optimizer = torch.optim.SGD(generator.parameters(), lr = Config['lr'])
        disc_optimizer = torch.optim.SGD(discriminator.parameters(), lr = Config['lr'])

    if Config['tensorboard_log']:
        writer = SummaryWriter(Config['checkpoint_path'])

    for epoch in tqdm(range(Config['epochs'])):
        generator.train()
        discriminator.train()
        for lr, hr in data_loader:
            valid = torch.zeros((lr.shape[0],1), requires_grad=False)
            fake = torch.ones((lr.shape[0],1), requires_grad=False)
            # print(lr.shape)
            sr = generator(lr)

            d_fake = discriminator(sr)
            d_real = discriminator(hr)

            c_loss = content_loss(args, feat_extractor, hr, sr)
            adv_loss = 1e-3 * nn.BCELoss()(valid, d_fake)
            mse_loss = nn.MSELoss()(hr, sr)
            perceptual_loss = c_loss + adv_loss + mse_loss

            valid_loss = nn.BCELoss()(valid, d_real)
            fake_loss = nn.BCELoss()(fake, d_fake)
            d_loss = valid_loss + fake_loss

            perceptual_loss.backward()
            d_loss.backward()

            gen_optimizer.step()
            disc_optimizer.step()
        generator.eval()
        discriminator.eval()
        test_lr, test_hr = next(iter(test_data_loader))
        with torch.set_grad_enabled(False):
            test_sr = generator(sr)
            for i in range(test_sr.shape[0]):
                img_sr = test_sr[i]
                img_hr = test_hr[i]
                img_lr = test_lr[i]
                save_image(img_sr, 'img_sr_%d.png'%i)
                save_image(img_hr, 'img_hr_%d.png'%i)
                save_image(img_lr, 'img_lr_%d.png'%i)

        print(f'Epoch {epoch}: Perceptual Loss:{perceptual_loss:.4f}, Disc Loss:{d_loss:.4f}')
    torch.save({'generator':generator,
                'discriminator':discriminator},
                os.path.join(Config['checkpoint_path'],'model.pth'))

if __name__=='__main__':
    train()
