import torch
from base import BaseVAE
from torch import nn
from typing import List, TypeVar
from torchvision.models import resnet34, resnet18
import torch.nn.functional as F

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

__all__ = ['AutoEncoder']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class AutoEncoder(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 512,
                 **kwargs) -> None:
        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim

        resnet = resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        modules.append(nn.Flatten(start_dim=1))
        modules.append(nn.Linear(resnet.fc.in_features, 512))
        modules.append(nn.BatchNorm1d(512, momentum=0.01))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(512, self.latent_dim))
        modules.append(nn.BatchNorm1d(self.latent_dim, momentum=0.01))
        modules.append(nn.ReLU(inplace=True))
        resnet = nn.Sequential(*modules)

        # self.encoder = nn.Sequential(*modules)
        self.encoder = resnet

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, 64*49)
        modules.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2), # 14
                    BasicBlock(64,64),
                    BasicBlock(64,64),
                    nn.UpsamplingBilinear2d(scale_factor=2), # 28
                    BasicBlock(64,64),
                    BasicBlock(64,64),
                    nn.UpsamplingBilinear2d(scale_factor=2), # 56
                    BasicBlock(64,64),
                    BasicBlock(64,64),

                    nn.UpsamplingBilinear2d(scale_factor=2), # 112
                    BasicBlock(64,64),
                    BasicBlock(64,64),
                    nn.Conv2d(64, 3*4, 3, padding="same"),
                    nn.PixelShuffle(2),
                    nn.sigmod()
                )
        )

        self.decoder = nn.Sequential(*modules)


    def forward(self, inputs: Tensor, **kwargs):
        vector = self.encoder(inputs)
        out = self.decoder_input(vector)
        out = out.view(-1, 64, 7, 7)
        out = self.decoder(out)
        return out, vector



if __name__ == '__main__':
    model = AutoEncoder(in_channels=3, latent_dim=128)
    x = torch.rand([3,3,224,224])
    out, vector = model(x)

    print(out.shape)
    print(vector.shape)

