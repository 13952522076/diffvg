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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", help="target image path")
    parser.add_argument("--save_folder", help="target image path")
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()
    files = [f for f in listdir(args.target_folder) if isfile(join(args.target_folder, f))]
    for file in files:
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            target_path = join(args.target_folder, file)
            save_path = join(args.save_folder, file)
            target = torch.from_numpy(skimage.io.imread(target_path)).to(torch.float32) / 255.0
            rendered = torch.from_numpy(skimage.io.imread(save_path)).to(torch.float32) / 255.0
            print(f"target shape {target.shape}, rendered shape {rendered.shape}")
