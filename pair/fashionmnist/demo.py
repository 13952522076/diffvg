import argparse
import os
import logging
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torchvision.datasets.mnist import FashionMNIST, MNIST
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_loader = DataLoader(FashionMNIST('../data', train=False, download=True, transform=transform),
                         num_workers=1, batch_size=4, shuffle=True, pin_memory=True)

data, label = next(iter(test_loader))
print(data.shape)
data = data[:16]
vutils.save_image(data, "view.png", nrow=8, normalize=True)
# print(data.shape)
#
# fig = plt.figure()
# for i in range(32):
#   plt.subplot(4,8,i+1)
#   plt.tight_layout()
#   img = (data[i][0]).numpy()
#
#   plt.imshow(img, cmap='gray', interpolation='none')
#   # plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
#   plt.show()
#
