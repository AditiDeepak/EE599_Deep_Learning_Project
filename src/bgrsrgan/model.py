import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Config
class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=2, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0),-1).std(dim=1).view(-1,1,1,1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.pointwise = nn.Conv2d(in_features, out_features, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_weight_norm=True):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Conv2d(in_channels, mid_channels, kernel_size=3, padding=1) if use_weight_norm else nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, k=3, n=64, s=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(in_features, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = DepthwiseSeparableConv2d(n,n)
        self.bn2 = nn.BatchNorm2d(n)
    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = y * F.sigmoid(y)
        return self.bn2(self.conv2(y)) + x

class UpsampleBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        y = self.shuffler(self.conv(x))
        return y*F.sigmoid(y)

class Generator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, bilinear=False):
        super(Generator, self).__init__()
        n_channels =[
            16,
            32,
            64,
            128
        ]
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128,256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024//factor)
        self.up1 = Up(1024//factor, 512//factor, bilinear)
        self.up2 = Up(512, 256//factor, bilinear)
        self.up3 = Up(256, 128//factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        self.out_bn = nn.BatchNorm2d(out_channels)

        self.n_residual_blocks = Config['n_resblocks']
        self.upsample_factor = Config['scale']
        self.conv1 = Conv2d(3, 64, 9, stride=1, padding=4, groups=1)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block'+str(i+1), ResidualBlock())

        self.conv2 = Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor//2):
            self.add_module('upsample'+str(i+1), UpsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, 9, stride=1, padding=4)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        res1 = self.outc(x)
        x = self.out_bn(res1)

        y = self.conv1(res1)
        x = y*F.sigmoid(y)

        y = x.clone()

        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block'+str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x
        for i in range(self.upsample_factor//2):
            x = self.__getattr__('upsample'+str(i+1))(x)

        return F.sigmoid(res1), self.conv3(x)

def get_feat_extractor():
    resnet = torchvision.models.resnet34(pretrained=True)
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
