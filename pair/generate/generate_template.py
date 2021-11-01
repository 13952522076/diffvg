"""
"""
import pydiffvg
import torch
import os
import numpy as np
import cv2
import skimage
import skimage.io
import matplotlib.pyplot as plt
import random
import argparse
import math
import errno
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import adaptive_avg_pool2d
import warnings
warnings.filterwarnings("ignore")
pydiffvg.set_print_timing(False)
gamma = 1.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_path", type=str, default="../data/generate/template")
    return parser.parse_args()

def main():
    args = parse_args()
    shapes_list = []
    shape_groups_list = []
    for root, dirs, files in os.walk(args.template_path):
        file_path = os.path.join(root, dirs, files)
        print(f"loading... {file_path}")
        # canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(os.path.join(args.template_path, file))
        # shapes_list.extend(shapes)
        # shape_groups_list.extend(shape_groups)




if __name__ == "__main__":
    main()
