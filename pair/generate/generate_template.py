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

    parser.add_argument("--path_num_min", type=int, default=1)
    parser.add_argument("--path_num_max", type=int, default=6)
    parser.add_argument("--generate_num", type=int, default=100)

    return parser.parse_args()

def main():
    args = parse_args()
    shapes_list = []
    shape_groups_list = []

    for root, dirs, files in os.walk(args.template_path):

        for file in files:
            if file.endswith(".svg"):
                file_path = os.path.join(root, file)

                canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(file_path)
                shapes_list.extend(shapes)
                shape_groups_list.extend(shape_groups)

    shapes_num = len(shapes_list)
    print(f"length of shapes_list is {len(shapes_list)}")

    for t in tqdm(range(args.generate_num)):
        path_num = np.random.randint(args.path_num_min, args.path_num_max+1)  # [path_num_min, path_num_max]
        path_indexes = np.random.randint(0, shapes_num, size=path_num).reshape((1, -1))
        selected_shapes = []
        selected_shape_groups = []
        for i in path_indexes:
            selected_shapes.append(shapes_list[i])
            selected_shape_groups.append(shape_groups_list[i])

        print(f"{len(selected_shapes)},   {len(selected_shape_groups)}")


if __name__ == "__main__":
    main()
