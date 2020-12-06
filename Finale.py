#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.image as mpimg
import os
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch
import torchvision
import random
from PIL import Image
import PIL.ImageOps
from torch.utils.tensorboard import SummaryWriter



#Performing filtering operation
def Salt_and_pepper_noise(image):
    count = 0
    lastMedian = image
    median = cv2.medianBlur(image, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, image))
        image[zeroed] = 0

        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(image, 3)
    return image

#find the significant contour
def Contour(image):
    contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)

    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

def resize(img):
    return cv2.resize(img,(240,320))

input_images = glob('/home/ec2-user/AWS-EE599/AWS/input_images/*.jpg')

for i in range(811,len(input_images)):
    print(i)
    break
    input_image = cv2.imread(input_images[i])
    input_image = resize(input_image)
    input_image = input_image[:,:,::-1]

    #perform gaussion blur
    blur = cv2.GaussianBlur(input_image, (5, 5), 0)
    blur = blur.astype(np.float32) / 255.0

    #use the model.yml file to perform edge detection (pre-trained)
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("/home/ec2-user/AWS-EE599/AWS/model.yml")
    edges = edgeDetector.detectEdges(blur) * 255.0
    edges_8u = np.asarray(edges, np.uint8)
    Salt_and_pepper_noise(edges_8u)

    contour = Contour(edges_8u)

    # Draw the contour on the original image
    contourImg = np.copy(input_image)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

    #Generate trimap
    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255

    # mask_path = "./mask_images/"
    trimap_path = "/home/ec2-user/AWS-EE599/AWS/trimap_images/"
    target_path = "/home/ec2-user/AWS-EE599/AWS/target_images/"

    try:
        os.stat(trimap_path)
    except:
        os.mkdir(trimap_path)

    try:
        os.stat(target_path)
    except:
        os.mkdir(target_path)


    cv2.imwrite(trimap_path + 'trimap_' + str(i) +'.jpg', trimap_print)

    # run grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    try:
        cv2.grabCut(input_image, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except:
        print("Grab-Cut Assertion error")
        continue
    # create mask again
    mask2 = np.where((trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),255,0).astype('uint8')
    # cv2.imwrite(mask_path + 'mask_' + str(i) +'.jpg', mask2)

    # estimate alpha from image and trimap
    alpha = mask2
    alpha = cv2.resize(alpha,(input_image.shape[1],input_image.shape[0]))
    alpha = alpha.astype(float)/255
    ones_for_alpha = np.ones((alpha.shape[0],alpha.shape[1]))



    # make gray background
    background_image = cv2.imread(random.choice(glob('/home/ec2-user/AWS-EE599/AWS/background_images/*.jpg')))
    background_image = cv2.resize(background_image, (input_image.shape[1],input_image.shape[0]), interpolation = cv2.INTER_AREA)
    background_image =  (ones_for_alpha - alpha)[:,:,np.newaxis] * background_image
    background_image = background_image.astype('uint8')
    # cv2.imwrite( 'back.jpg', background_image)


    # estimate foreground from image and alpha
    foreground = (input_image * alpha[:,:,np.newaxis])[:,:,::-1]
    foreground = foreground.astype('uint8')
    # cv2.imwrite( 'front.jpg', foreground)

    # blend foreground with background and alpha, less color bleeding
    out_image = cv2.add(foreground, background_image)
    cv2.imwrite(target_path + 'target_' + str(i) +'.jpg', out_image)

print("Done loading images...!")

# Data Loader Class

Config={}
Config['num_epochs']=50
Config['batch_size']=100
Config['learning_rate']=0.001
Config['disc_loss_coeff']=1.0
Config['gen_model_path']='/home/ec2-user/AWS-EE599/AWS/generator/'
Config['disc_model_path']='/home/ec2-user/AWS-EE599/AWS/discriminator/'

if not os.path.exists('/home/ec2-user/AWS-EE599/AWS/generator/'):
    os.makedirs('/home/ec2-user/AWS-EE599/AWS/generator/')
if not os.path.exists('/home/ec2-user/AWS-EE599/AWS/discriminator/'):
    os.makedirs('/home/ec2-user/AWS-EE599/AWS/discriminator/')


input_path = '/home/ec2-user/AWS-EE599/AWS/input_images/'
trimap_path = '/home/ec2-user/AWS-EE599/AWS/trimap_images/'
target_path = '/home/ec2-user/AWS-EE599/AWS/target_images/'

class Background_Removal_Dataset(torch.utils.data.Dataset):
    def __init__(self, input_path,trimap_path,target_path):

        self.input_path = input_path
        self.trimap_path = trimap_path
        self.target_path = target_path
        self.input_image = os.listdir(input_path)
        self.trimap_image = os.listdir(trimap_path)
        self.target_image = os.listdir(target_path)

        self.transform = transforms.Compose([transforms.Resize((320,240)),transforms.ToTensor()])

    def __len__(self):
        return len(self.input_image)

    def __getitem__(self, idx):
        input_image = Image.open(os.path.join(self.input_path, self.input_image[idx]))
        trimap_image = Image.open(os.path.join(self.trimap_path, self.trimap_image[idx]))
        target_image = Image.open(os.path.join(self.target_path, self.target_image[idx])).convert('RGBA')

        return torch.cat((self.transform(input_image), self.transform(trimap_image)),0), self.transform(target_image)

dataset = Background_Removal_Dataset(input_path,trimap_path,target_path)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=Config['batch_size'], shuffle = True)
print("Dataloader Ready..!")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Discriminator(nn.Module):
    def __init__(self, pretrained=False):
        super(Discriminator, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
    def forward(self, x):
        return F.sigmoid(self.resnet(x))


# Train Script

def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)

def train():
    generator=UNet(4,4)
    discriminator=Discriminator(4)

    gen_loss=nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    opt_generator = torch.optim.Adadelta(generator.parameters(), lr=Config['learning_rate'])
    opt_generator2 = torch.optim.Adadelta(generator.parameters(), lr=Config['learning_rate'])
    opt_discriminator = torch.optim.Adadelta(discriminator.parameters(), lr=Config['learning_rate'])

    writer = SummaryWriter()
    for i in range(Config['num_epochs']):
        print("Epoch:",i,"running..!")
        mean_gen_loss=0.0
        mean_disc_loss=0.0
        mean_gen_total_loss=0.0

        for inputs,target in train_loader:
            #Inputs consists of (image,trimap)
            #Output consists of the 'target image'
            print("Starting Training...")
            inputs = torch.autograd.Variable(inputs.to(device))
            target = torch.autograd.Variable(target.to(device))

            #For generator

            output_gen=torch.sigmoid(generator(inputs))
            gen_loss_batch=gen_loss(output_gen,target)
            mean_gen_loss+=gen_loss_batch.data

            print('Epoch: MSE Loss',gen_loss_batch.data)

            #For discriminator

            output_disc_real=discriminator(target)
            output_disc_fake=discriminator(output_gen)
            disc_loss = wasserstein_loss(output_disc_real,output_disc_fake)

            mean_disc_loss+=disc_loss.data

            opt_discriminator.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_discriminator.step()
            print("One done.")

            total_loss = gen_loss_batch + Config['disc_loss_coeff'] * disc_loss
            mean_gen_total_loss+=total_loss
            opt_generator2.zero_grad()
            total_loss.backward()
            opt_generator2.step()

        #loss per epoch
        writer.add_scalar('discriminator_loss', mean_disc_loss/len(train_loader), global_step=i)
        writer.add_scalar('generator_loss', mean_gen_loss/len(train_loader), global_step=i)
        writer.add_scalar('total_generator_loss', mean_gen_total_loss/len(train_loader), global_step=i)

    print("Done Training..!")
    torch.save(generator.state_dict(), Config['gen_model_path'], 'generator_final.pth')
    torch.save(discriminator.state_dict(), Config['disc_model_path'], 'discriminator_final.pth')

def main():
    train()
    return 0

if __name__ == '__main__':
    main()



