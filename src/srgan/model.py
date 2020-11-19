import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import math
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0,1.0,1.0), sign=-1):
        super(MeanShift, self).__init__()

        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3,3,1,1)/std.view(3,1,1,1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)/std
        for p in self.parameters():
            p.requires_grad=False

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m=[]
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i==0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class UpSampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m=[]
        if (scale & scale-1)==0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4*n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act=='relu':
                    m.append(nn.ReLU(True))
                elif act=='prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale==3:
            m.append(conv(n_feats, 9*n_feats, 3, bias))
            m.append(nn.PixelShuffle(3)))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act=='relu':
                m.append(nn.ReLU(True))
            elif act=='prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(UpSampler, self).__init__(*m)

class EDSR_Generator(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR_Generator, self).__init__()

        n_resblocks = args['n_resblocks']
        n_feats = args['n_feats']
        kernel_size = 3
        scale = args['scale'][0]
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(args['rgb_range'])
        self.add_mean = MeanShift(args['rgb_range'], sign=1)

        self.head = nn.Sequential([conv(args['n_colors'], n_feats, kernel_size)])
        self.body = nn.Sequential([
                        ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args['res_scale'])
                        for _ in range(n_resblocks)].append(conv(n_feats, n_feats, kernel_size)))
        self.tail = nn.Sequential([
                                UpSampler(conv, int(scale), n_feats, act=False),
                                conv(n_feats, args['n_colors'], kernel_size)
                            ])
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

# class EDSR_Discriminator(nn.Module):
#     def __init__(self, pretrained=False):
#         self.model = models.mobilenet_v2(pretrained=pretrained)
#         self.model.classifier[1] = torch.nn.Linear(in_features = self.model.classifier[1].in_features, 1)
#     def forward(self, x):
#         return F.sigmoid(self.model(x))


class Discriminator(nn.Module):
    '''
        output is not normalized
    '''
    def __init__(self, args):
        super(Discriminator, self).__init__()

        in_channels = args['n_colors']
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))

        patch_size = args['patch_size'] // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

def content_loss(self, args, vgg, hr, sr):
    sub_mean = MeanShift(args['rgb_range'])
    sr = sub_mean(sr)
    sr_features = vgg(sr)

    hr = sub_mean(hr)
    hr_features = vgg(hr)
    return nn.MSELoss()(sr_features, hr_features)

def adversarial_loss(self, d_fake, d_real):
    loss_d = (d_fake - d_real).mean()
    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
    hat.requires_grad = True
    d_hat = self.dis(hat)
    gradients = torch.autograd.grad(
        outputs=d_hat.sum(), inputs=hat,
        retain_graph=True, create_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
    loss_d += gradient_penalty

    loss_g = -d_fake.mean()

    return loss_d, loss_g
