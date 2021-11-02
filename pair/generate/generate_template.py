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
    parser.add_argument("--generate_path", type=str, default="../data/generate/generate")
    parser.add_argument("--path_num_min", type=int, default=1)
    parser.add_argument("--path_num_max", type=int, default=8)
    parser.add_argument("--generate_num", type=int, default=2000)

    return parser.parse_args()

def main():
    args = parse_args()
    shapes_list = []
    shape_groups_list = []

    try:
        os.makedirs(os.path.join(args.generate_path,'img'))
    except OSError as exc:  # Python >2.5
        pass
    try:
        os.makedirs(os.path.join(args.generate_path,'svg'))
    except OSError as exc:  # Python >2.5
        pass

    for root, dirs, files in os.walk(args.template_path):

        for file in files:
            if file.endswith(".svg"):
                file_path = os.path.join(root, file)

                canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(file_path)
                canvas_width, canvas_height = 240, 240
                shapes_list.extend(shapes)
                shape_groups_list.extend(shape_groups)

    shapes_num = len(shapes_list)
    print(f"length of shapes_list is {len(shapes_list)}")

    for t in tqdm(range(args.generate_num)):
        path_num = np.random.randint(args.path_num_min, args.path_num_max+1)  # [path_num_min, path_num_max]
        path_indexes = np.random.randint(0, shapes_num, size=path_num).tolist()
        selected_shapes = []
        selected_shape_groups = []
        shape_groups = []
        shapes = []
        for i in path_indexes:
            selected_shape = shapes_list[i]
            selected_shape_group = shape_groups_list[i]

            new_path = pydiffvg.Path(num_control_points = selected_shape.num_control_points,
                                     points = selected_shape.points,
                                     stroke_width = torch.tensor(1.0),
                                     is_closed = True)
            points = new_path.points
            points = points + 1*(torch.rand_like(points)-0.5)
            # if np.random.randint(1,3) ==1: # random shift the position
            #     mean_point = new_path.points.mean(dim=1, keepdim=True)
            #     points = points - mean_point + torch.rand_like(mean_point)*1.1*mean_point
            # if np.random.randint(1,3) ==1: # random add some disturbance
            #     points = points * (1+ (0.02*(torch.rand_like(points)-0.5)))  # [0.85-1.05]
            new_path.points = points
            shapes.append(new_path)

            # process color
            color = selected_shape_group.fill_color
            if isinstance(selected_shape_group.fill_color, pydiffvg.RadialGradient):
                print(f"{t} includes RadialGradient")
                color.center = torch.rand_like(color.center)*(torch.tensor([canvas_width, canvas_height]).to(color.center.device))
                color.radius = torch.rand_like(color.radius)*(torch.tensor([canvas_width, canvas_height]).to(color.radius.device))
                # color.center = color.center * (0.1*(torch.rand_like(color.center)-0.5)+1.0)
                # color.radius = color.radius * (0.1*(torch.rand_like(color.radius)-0.5)+1.0)
                color.stop_colors = torch.rand_like(color.stop_colors)*1.3-0.1
                color.stop_colors[:,3] = color.stop_colors[:,3]*5  # make most are 1.0
                color.stop_colors.data.clamp_(0.0, 1.0)
            elif isinstance(selected_shape_group.fill_color, pydiffvg.LinearGradient):
                print(f"{t} includes LinearGradient")
                color.begin = torch.rand_like(color.begin)*(torch.tensor([canvas_width, canvas_height]).to(color.begin.device))
                color.end = torch.rand_like(color.end)*(torch.tensor([canvas_width, canvas_height]).to(color.end.device))
                color.begin[0] = min(color.begin[0],color.end[0] )
                color.begin[1] = min(color.begin[1],color.end[1] )
                color.end[0] = max(color.begin[0],color.end[0] )
                color.end[1] = max(color.begin[1],color.end[1] )
                # color.begin = color.begin * (0.1*(torch.rand_like(color.begin)-0.5)+1.0)
                # color.end = color.end * (0.1*(torch.rand_like(color.end)-0.5)+1.0)
                color.stop_colors = torch.rand_like(color.stop_colors)*1.3-0.1
                color.stop_colors[:,3] = color.stop_colors[:,3]*5  # make most are 1.0
                color.stop_colors.data.clamp_(0.0, 1.0)
            else:
                color = torch.rand_like(color)*1.3-0.1
                color[3] = color[3]*5  # make most are 1.0
                color.data.clamp_(0.0, 1.0)

            new_path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                         fill_color = color)
            shape_groups.append(new_path_group)

        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
        if np.random.randint(1,3) ==1: # random add background
            # Compose img with white background
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        else:
            img = img[:, :, 3:4] * img[:, :, :3] + torch.rand(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        pydiffvg.imwrite(img.cpu(), os.path.join(args.generate_path, "img", str(t)+'.png'), gamma=gamma)
        pydiffvg.save_svg(os.path.join(args.generate_path, "svg", str(t)+'.svg'),
                          canvas_width, canvas_height, shapes, shape_groups)

if __name__ == "__main__":
    main()
