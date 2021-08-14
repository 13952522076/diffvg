#!/bin/env python
"""Train a AE Fashion-MNIST.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydiffvg
from torchvision.models import resnet50
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pydiffvg.set_use_gpu(torch.cuda.is_available())


class Predictor(nn.Module):
    def __init__(self, paths=512, segments=2, max_width=2.0, im_size=224.0):
        super(Predictor, self).__init__()
        self.points = nn.Parameter(torch.rand(2 * paths * (segments * 3 + 1)))
        self.widths = nn.Parameter(torch.rand(paths))
        self.colors = nn.Parameter(torch.rand(paths*4))
        self.max_width = max_width
        self.im_size = im_size

    def forward(self):  # [b,z_dim]
        points = torch.tanh(self.points)
        points = points * (self.im_size // 2) + self.im_size // 2
        widths = torch.sigmoid(self.widths)
        widths = (self.max_width - 1) * widths + 1
        colors = torch.sigmoid(self.colors)
        return {
            "points": points,
            "widths": widths,
            "colors": colors
        }



if __name__ == '__main__':

    model = Predictor()
    model.to("cuda")
    out = model()
    print(out)

