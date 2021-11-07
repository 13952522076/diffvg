"""
Numerically interpolate,
python interpolate.py 1.svg 2.svg --alpha 0.1
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
    parser.add_argument("target1", help="target image path")
    parser.add_argument("target2", help="target image path")
    parser.add_argument('--alpha',  type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()
    # Load SVG
    canvas_width_1, canvas_height_1, shapes_1, shape_groups_1 = pydiffvg.svg_to_scene(args.target1)
    canvas_width_2, canvas_height_2, shapes_2, shape_groups_2 = pydiffvg.svg_to_scene(args.target2)
    assert canvas_width_1 == canvas_width_2 and \
           canvas_height_1 ==canvas_height_2 and len(shapes_1)==len(shapes_2), \
        "Please ensure two svg has same number of paths, and some resolution"

    filename1 = os.path.splitext(os.path.basename(args.target1))[0]
    filename2 = os.path.splitext(os.path.basename(args.target2))[0]
    new_shapes = []
    new_shape_groups = []
    for i in range(0, len(shape_groups_1)):
        shape1, shape2 = shapes_1[i], shapes_2[i]
        shape_group1, shape_group2 = shape_groups_1[i],shape_groups_2[i]
        points = args.alpha*shape1.points + (1.0-args.alpha)*shape2.points
        path = pydiffvg.Path(num_control_points = shape1.num_control_points,
                             points = points,
                             stroke_width = torch.tensor(1.0),
                             is_closed = True)
        color = args.alpha*shape_group1.fill_color + (1.0-args.alpha)*shape_group2.fill_color
        path_group = pydiffvg.ShapeGroup(shape_ids = shape_group1.shape_ids,
                                         fill_color = color)
        new_shapes.append(path)
        new_shape_groups.append(path_group)
    pydiffvg.save_svg(f"{filename1}-{filename2}-{str(args.alpha)}.svg",
                      canvas_width_1, canvas_height_1, new_shapes, new_shape_groups)

if __name__ == "__main__":
    main()
