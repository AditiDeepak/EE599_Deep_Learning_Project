import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Discriminator, get_feat_extractor

import os
import random
from utils import Config, PSNR, SSIM
import matplotlib.pyplot as plt

def visualize(gen_inputs, real_inputs, fake_inputs):
    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                                                                std=[4.367,4.464,4.444]),
                                            torchvision.transforms.ToPILImage(),
                                            torchvision.transforms.Scale(Config['img_size'])
                                            ])
    fig, (lr_plot, hr_plot, sr_plot) = plt.subplots(1,3)

    i = random.randint(0, gen_inputs.size(0)-1)
    lr_image = transform(gen_inputs[i])
    hr_image =  transform(real_inputs[i])
    sr_image = transform(fake_inputs[i])

    lr_plot.imshow(lr_image)
    hr_plot.imshow(hr_image)
    sr_plot.imshow(sr_image)

    return fig

def train():
    if not os.path.exists(Config['checkpoint_path']):
        os.makedirs(Config['checkpoint_path'])
        os.makedirs(os.path.join(Config['checkpoint_path'],'generators'))
        os.makedirs(os.path.join(Config['checkpoint_path'],'discriminators'))

    device=torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

    # print(Config['img_size']*Config['scale'])
    transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop((Config['img_size'][0]*Config['scale'],Config['img_size'][1]*Config['scale'])),
                                torchvision.transforms.ToTensor()
                                ])

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
    #print(Config['img_size'])
    scale = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.Resize(Config['img_size']),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std = [0.229, 0.224, 0.225])
                                                            ])
    dataset = torchvision.datasets.ImageFolder(root = Config['train_set_path'], transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=Config['batch_size'], shuffle=True, num_workers=Config['num_workers'])

    generator = Generator(Config)
    if Config['generator_checkpoint']:
        generator.load_state_dict(torch.load(Config['generator_checkpoint']))

    discriminator = Discriminator(Config)
    if Config['discriminator_checkpoint']:
        discriminator.load_state_dict(torch.load(Config['discriminator_checkpoint']))

    feat_extractor = get_feat_extractor()

    content_loss = nn.MSELoss()
    adversarial_loss = nn.BCELoss()

    ones_const = torch.autograd.Variable(torch.ones(Config['batch_size'],1))

    generator.to(device)
    discriminator.to(device)
    feat_extractor.to('cpu')
    ones_const.to(device)

    opt_generator = torch.optim.Adam(generator.parameters(), lr=Config['generator_lr'])
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=Config['discriminator_lr'])

    if Config['tensorboard_log']:
        writer_pretrain = SummaryWriter(os.path.join(Config['checkpoint_path'],'pretrain'))
        writer = SummaryWriter(Config['checkpoint_path'])

    low_res = torch.FloatTensor(Config['batch_size'], 3, Config['img_size'][0], Config['img_size'][1])

    for epoch in range(2):

        mean_generator_content_loss = 0.0

        for i, data in enumerate(data_loader):
            high_res_real, _ = data

            if high_res_real.shape[0]!=Config['batch_size']: continue
            for j in range(Config['batch_size']):
                low_res[j]=scale(high_res_real[j])
                high_res_real[j]=normalize(high_res_real[j])

            high_res_real = torch.autograd.Variable(high_res_real.to(device))
            high_res_fake = generator(torch.autograd.Variable(low_res).to(device))

            generator.zero_grad()
            #print(high_res_fake.shape, high_res_real.shape)
            generator_content_loss = content_loss(high_res_fake, high_res_real)
            #print(generator_content_loss)
            mean_generator_content_loss += generator_content_loss.data

            generator_content_loss.backward()
            opt_generator.step()
            print(f'Epoch {epoch} Iter {i/len(data_loader)}: MSE Loss {generator_content_loss.data}')
            writer.add_figure('Pretrain SR Generator',visualize(low_res, high_res_real.cpu().data, high_res_fake.cpu().data), global_step = epoch)
        writer.add_scalar('generator_mse_loss', mean_generator_content_loss/len(data_loader), epoch)
    torch.save(generator.state_dict(), os.path.join(Config['checkpoint_path'],'generators/generator_pretrain.pth'))

    opt_generator = torch.optim.Adam(generator.parameters(), lr=Config['generator_lr']*0.1)
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=Config['discriminator_lr']*0.1)

    for epoch in range(Config['epochs']):
        mean_generator_content_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0

        for i, data in enumerate(data_loader):
            high_res_real, _ = data
            psnrs = []
            ssims = []

            for j in range(Config['batch_size']):
                low_res[j] = scale(high_res_real[j])
                high_res_real[j] = normalize(high_res_real[j])

            low_res.to(device)
            high_res_real = torch.autograd.Variable(high_res_real.to(device))
            target_real = torch.autograd.Variable(torch.rand(Config['batch_size'],1)*0.5 +0.7).to(device)
            target_fake = torch.autograd.Variable(torch.rand(Config['batch_size'],1)*0.3).to(device)
            high_res_fake = generator(torch.autograd.Variable(low_res).to(device))

            discriminator.zero_grad()
            discriminator_loss = adversarial_loss(discriminator(high_res_real), target_real) + adversarial_loss(discriminator(high_res_fake), target_fake)

            mean_discriminator_loss += discriminator_loss.data

            discriminator_loss.backward(retain_graph=True)
            opt_discriminator.step()

            generator.zero_grad()

            real_features = torch.autograd.Variable(feat_extractor(high_res_real.to('cpu')).to(device).data)
            fake_features = feat_extractor(high_res_fake.to('cpu')).to(device)

            generator_content_loss = content_loss(high_res_fake, high_res_real) + 0.006 * content_loss(fake_features, real_features)
            mean_generator_content_loss += generator_content_loss.data

            generator_adversarial_loss = adversarial_loss(discriminator(high_res_fake), ones_const.to(device))
            mean_generator_adversarial_loss += generator_adversarial_loss.data

            generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss.data

            generator_total_loss.backward()
            opt_generator.step()

            for j in range(Config['batch_size']):
                psnrs.append(PSNR()(high_res_real[j], high_res_fake[j]))
                ssims.append(SSIM()(high_res_real[j], high_res_fake[j]))

            print(f'Epoch {epoch} Iter {i/len(data_loader)}: Discriminator Loss {discriminator_loss.data}, Generator Loss: {generator_content_loss.data}/{generator_adversarial_loss.data}/{generator_total_loss.data}')
            writer.add_figure('Training Super Resolution',visualize(low_res, high_res_real.cpu().data, high_res_fake.cpu().data), global_step = epoch)
        psnr = np.array(psnrs).mean()
        ssim = np.array(ssims).mean()
        writer.add_scalar('discriminator_loss', mean_discriminator_loss/len(data_loader), global_step=epoch)
        writer.add_scalar('generator_content_loss', mean_generator_content_loss/len(data_loader), global_step=epoch)
        writer.add_scalar('generator_adversarial_loss', mean_generator_adversarial_loss/len(data_loader), global_step=epoch)
        writer.add_scalar('generator_total_loss', mean_generator_total_loss/len(data_loader), global_step=epoch)
        writer.add_scalar('PSNR', psnr, global_step=epoch)
        writer.add_scalar('SSIM', ssim, global_step=epoch)
    torch.save(generator.state_dict(), os.path.join(Config['checkpoint_path'], 'generators/generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(Config['checkpoint_path'], 'discriminators/discriminator_final.pth'))

if __name__=='__main__':
    train()
