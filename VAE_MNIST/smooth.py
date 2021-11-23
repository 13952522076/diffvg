from __future__ import print_function
import argparse
import torch
# import cv2
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
import os
try:
    os.mkdir("hr_binary")
except:
    pass
filename = "id1_id4_id30_id60_1.0_1.0.png"
image = Image.open(f'hr/{filename}')
x = TF.to_tensor(image)
x.unsqueeze_(0)
x[x>=0.4] = 1.0
x[x<0.4] = 0.0
# x = TF.resize(x, 56)
x = TF.resize(x,224, interpolation=TF.InterpolationMode.BILINEAR)

save_image(x, f'hr_binary/{filename}')



print(x.shape)
