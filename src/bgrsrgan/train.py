import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import Generator, Discriminator, get_feat_extractor
from utils import Config, PSNR, SSIM
from data import BGRSRDataset

import os
import random
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim

def visualize(img_inputs, bg_inputs, repl_outputs_lr, repl_outputs_sr, repl_outputs_hr):
    transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                                                                std=[4.367,4.464,4.444]),
                                            torchvision.transforms.ToPILImage(),
                                            torchvision.transforms.Scale(Config['img_size'])
                                            ])
    fig, (img_plot, bg_plot, repl_oplr_plot, repl_opsr_plot, repl_ophr_plot) = plt.subplots(1,5)

    i = random.randint(0, img_inputs.size(0)-1)
    img = transform(img_inputs[i])
    bg = transform(bg_inputs[i])
    lr_image = transform(repl_outputs_lr[i])
    hr_image =  transform(repl_outputs_hr[i])
    sr_image = transform(repl_outputs_sr[i])

    img_plot.imshow(img)
    bg_plot.imshow(img)
    repl_oplr_plot.imshow(lr_image)
    repl_ophr_plot.imshow(hr_image)
    repl_opsr_plot.imshow(sr_image)

    return fig

def train():
    if not os.path.exists(Config['checkpoint_path']):
        os.makedirs(Config['checkpoint_path'])
        os.makedirs(os.path.join(Config['checkpoint_path'],'generators'))
        os.makedirs(os.path.join(Config['checkpoint_path'],'discriminators'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = BGRSRDataset(os.path.join(Config['train_set_path'],'images'),os.path.join(Config['train_set_path'],'backgrounds'))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=Config['batch_size'], shuffle=True, num_workers=Config['num_workers'])

    generator = Generator()
    if Config['generator_checkpoint']:
        generator.load_state_dict(torch.load(Config['generator_checkpoint']))

    discriminator = Discriminator()
    if Config['discriminator_checkpoint']:
        discriminator.load_state_dict(torch.load(Config['discriminator_checkpoint']))

    feat_extractor = get_feat_extractor()

    background_replacement_loss = nn.MSELoss()
    content_loss = nn.MSELoss()
    adversarial_loss = nn.BCELoss()

    ones_const = torch.autograd.Variable(torch.ones(Config['batch_size'],1))

    generator.to(device)
    discriminator.to(device)
    feat_extractor.to('cpu')
    ones_const.to(device)

    opt_generator = torch.optim.Adam(generator.parameters(), lr=Config['generator_lr'])
    # opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=Config['discriminator_lr'])

    if Config['tensorboard_log']:
        writer_pretrain = SummaryWriter(os.path.join(Config['checkpoint_path'], 'pretrain'))
        writer = SummaryWriter(Config['checkpoint_path'])

    low_res = torch.FloatTensor(Config['batch_size'], 6, Config['img_size'][0], Config['img_size'][1])
    for epoch in range(int(Config['batch_size']*0.3)):
        mean_generator_content_loss = 0.0
        mean_background_replacement_loss = 0.0
        for i, data in enumerate(data_loader):
            input_images, background_images, bgrep_images, output_images = data

            if input_images.shape[0]!=Config['batch_size']: continue
            inputs = torch.cat([input_images, background_images],1)

            high_res_real = torch.autograd.Variable(output_images.to(device))
            low_res_fake, high_res_fake = generator(torch.autograd.Variable(low_res.to(device)))

            generator.zero_grad()

            bgr_content_loss = background_replacement_loss(low_res_fake, bgrep_images)
            gen_content_loss = content_loss(high_res_fake, output_images)
            total_gen_loss = bgr_content_loss + gen_content_loss

            mean_generator_content_loss += gen_content_loss.data
            mean_background_replacement_loss += bgr_content_loss.data

            total_gen_loss.backward()
            opt_generator.step()

            print(f'Epoch {epoch} Iter {i,len(data_loader)}: BGR Loss:{bgr_content_loss.data}, SR Loss:{gen_content_loss.data}')
            writer.add_figure('Pretrain BGRSR Generator',visualize(input_images, background_images, low_res_fake, high_res_fake, output_images), global_step=epoch)
        writer.add_scalar('background_replacement_loss', mean_generator_content_loss/len(data_loader), epoch)
        writer.add_scalar('sr_content_loss', mean_generator_content_loss/len(data_loader), epoch)
    torch.save(generator.state_dict(), os.path.join(Config['checkpoint_path'],'generators/generator_pretrain.pth'))

    opt_generator = torch.optim.Adam(generator.parameters(), lr=Config['generator_lr'])
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=Config['discriminator_lr'])



    for epoch in range(Config['epochs']):
        mean_generator_replacement_loss = 0.0
        mean_generator_content_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0

        for i, data in enumerate(data_loader):
            input_images, background_images, bgrep_images, output_images = data
            psnrs = []
            ssims = []

            output_images = torch.autograd.Variable(output_images.to(device))
            target_real = torch.autograd.Variable(torch.rand(Config['batch_size'],1)*0.5 +0.7).to(device)
            target_fake = torch.autograd.Variable(torch.rand(Config['batch_size'],1)*0.3).to(device)

            inputs = torch.cat([input_images, background_images],1)
            low_res_fake, high_res_fake = generator(torch.autograd.Variable(inputs.to(device)))

            discriminator.zero_grad()
            discriminator_loss = adversarial_loss(discriminator(output_images), target_real) + adversarial_loss(discriminator(high_res_fake), target_fake)

            mean_discriminator_loss += discriminator_loss.data

            discriminator_loss.backward(retain_graph=True)
            opt_discriminator.step()

            generator.zero_grad()

            real_features = torch.autograd.Variable(feat_extractor(output_images.to('cpu')).to(device).data)
            fake_features = feat_extractor(high_res_fake.to('cpu')).to(device)

            generator_content_loss = content_loss(high_res_fake, output_images) + 0.006 * content_loss(fake_features, real_features)
            mean_generator_content_loss += generator_content_loss.data

            real_bgrep_features = torch.autograd.Variable(feat_extractor(bgrep_images.to('cpu')).to(device).data)
            fake_bgrep_features = feat_extractor(low_res_fake.to('cpu')).to(device)

            bgr_content_loss = background_replacement_loss(low_res_fake, bgrep_images) + 0.006 * content_loss(fake_bgrep_features, real_bgrep_features)
            mean_background_replacement_loss += bgr_content_loss.to(device)

            generator_adversarial_loss = adversarial_loss(discriminator(high_res_fake), ones_const.to(device))
            mean_generator_adversarial_loss += generator_adversarial_loss.data

            generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss

            mean_generator_total_loss += generator_total_loss.data

            generator_total_loss.backward()
            opt_generator.step()

            for j in range(Config['batch_size']):
                img1 = output_images[j].numpy().transpose(1,2,0)
                img2 = high_res_fake[j].numpy().transpose(1,2,0)
                psnrs.append(PSNR()(img1, img2))
                ssims.append(compare_ssim(img1,img2, gradient=False))

            print(f'Epoch {epoch} Iter {i/len(data_loader)}: Discriminator Loss {discriminator_loss.data}, BG Replacement Loss: {background_replacement_loss.data/generator_content_loss.data/generator_adversarial_loss.data/generator_total_loss.data}')
            writer.add_figure('Training BGRSRGAN', visualize(input_images, background_images, low_res_fake, high_res_fake, output_images), epoch)
        psnr = np.array(psnrs).mean()
        ssim = np.array(ssims).mean()
        writer.add_scalar('discriminator_loss', mean_discriminator_loss/len(data_loader), epoch)
        writer.add_scalar('background_replacement_loss', mean_background_replacement_loss/len(data_loader), epoch)
        writer.add_scalar('generator_content_loss', mean_generator_content_loss/len(data_loader), epoch)
        writer.add_scalar('generator_adversarial_loss', mean_generator_adversarial_loss/len(data_loader), epoch)
        writer.add_scalar('generator_total_loss', generator_total_loss/len(data_loader), epoch)
        writer.add_scalar('PSNR', psnr, epoch)
        writer.add_scalar('SSIM', ssim, epoch)

        if (epoch+1)%5 == 0:
            torch.save(generator.state_dict, os.path.join(Config['checkpoint_path'],'generators/generator_'+str(epoch+1)+'.pth'))
            torch.save(discriminator.state_dict, os.path.join(Config['checkpoint_path'],'discriminators/discriminator_'+str(epoch+1)+'.pth'))
    torch.save(generator.state_dict, os.path.join(Config['checkpoint_path'],'generators/generator_'+str(epoch+1)+'.pth'))
    torch.save(discriminator.state_dict, os.path.join(Config['checkpoint_path'],'discriminators/discriminator_'+str(epoch+1)+'.pth'))

if __name__=='__main__':
    train()
