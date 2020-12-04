import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG, make_layers, cfgs, vgg19_bn
import torchvision

from utils import Config

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.pointwise = nn.Conv2d(in_features, out_features, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, k=3, n=64, s=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_features, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = DepthwiseSeparableConv2d(n, n)
        self.bn2 = nn.BatchNorm2d(n)
    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = y * F.sigmoid(y)
        return self.bn2(self.conv2(y)) + x

class UpsampleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpsampleBlock, self).__init__()
        # self.usampler = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Conv2d(in_features, out_features, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        y = self.shuffler(self.conv(x))
        return y * F.sigmoid(y)

class Generator(nn.Module):
    def __init__(self, args=Config):
        super(Generator, self).__init__()
        self.n_residual_blocks = args['n_resblocks']
        self.upsample_factor = args['scale']

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block'+str(i+1), ResidualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample'+str(i+1), UpsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        y = self.conv1(x)
        x = y*F.sigmoid(y)

        y = x.clone()

        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block'+str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample'+str(i+1))(x)

        return self.conv3(x)

# class FeatExtractor(VGG):
#     def __init__(self):
#         super(FeatExtractor, self).__init__(make_layers(cfgs['D']))
#     def forward(self, x):
#         return torch.flatten(self.avgpool(self.features(x)),1)

def get_feat_extractor():
    resnet = torchvision.models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad=False
    return resnet

class Discriminator(nn.Module):
    def __init__(self, pretrained=False):
        super(Discriminator, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
    def forward(self, x):
        return F.sigmoid(self.resnet(x))
