import torch
import warnings
warnings.filterwarnings("ignore")
import PIL
import PIL.Image
import numpy as np
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--target", default="images/emoji/demo2.png", help="target image path")
parser.add_argument("--render", default="images/emoji/LIVE/1,1,1,1,4.png", help="target image path")
parser.add_argument("--filename", type=str)
args = parser.parse_args()
target = torch.from_numpy(np.array(PIL.Image.open(args.target))).to(torch.float32) / 255.0
if len(target.shape) == 2:
    target = target.unsqueeze(dim=-1).repeat(1,1,3)
if target.shape[2] == 4:
    target = target[:, :, :3]
render = torch.from_numpy(np.array(PIL.Image.open(args.render))).to(torch.float32) / 255.0
if len(render.shape) == 2:
    render = render.unsqueeze(dim=-1).repeat(1,1,3)
if render.shape[2] == 4:
    render = render[:, :, :3]
print(target.shape, render.shape)
target = target.permute(2,0,1).unsqueeze(dim=0)
render = render.permute(2,0,1).unsqueeze(dim=0)
render = F.interpolate(render, (240,240))
loss = F.mse_loss(render, target)
print(loss)
