"""
python mse.py --target_folder data/emoji_rgb/all --save_folder ../pair/Layerwise/evaluate/emoji_path8
pair_vgtest240
python mse.py --target_folder data/pair_vgtest240 --save_folder ../pair/Layerwise/evaluate/pair_vgtest240_path8
"""
import pydiffvg
import torch
import os
from os import listdir
from os.path import isfile, join
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", help="target image path")
    parser.add_argument("--save_folder", help="target image path")
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()
    files = [f for f in listdir(args.target_folder) if isfile(join(args.target_folder, f))]
    targets = []
    renders = []

    for file in files:
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            target_path = join(args.target_folder, file)
            save_path = join(args.save_folder, file)
            target = torch.from_numpy(skimage.io.imread(target_path)).to(torch.float32) / 255.0
            if len(target.shape)==2:
                # print("Converting the gray-scale image to RGB.")
                target = target.unsqueeze(dim=-1).repeat(1,1,3)
            if target.shape[2] == 4:
                # print("Input image includes alpha channel, simply dropout alpha channel.")
                target = target[:, :, :3]
            render = torch.from_numpy(skimage.io.imread(save_path)).to(torch.float32) / 255.0
            print(f"target shape {target.shape}, rendered shape {render.shape}")
            target = target.permute(2,0,1).unsqueeze(dim=0)
            target = F.interpolate(target, (240,240))
            render = render.permute(2,0,1).unsqueeze(dim=0)
            render = F.interpolate(render, (240,240))
            targets.append(target)
            renders.append(render)

    targets = torch.cat(targets,dim=0)
    renders = torch.cat(renders,dim=0)
    loss = F.mse_loss(renders, targets)
    print(f"loss is {loss}")
