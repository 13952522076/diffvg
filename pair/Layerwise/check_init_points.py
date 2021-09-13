"""
python check_init_points.py demo.png --num_paths 1,1,1,1,1
"""
import pydiffvg
import torch
import os
import matplotlib.pyplot as plt
import random
import argparse
import math
import errno
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import adaptive_avg_pool2d

pydiffvg.set_print_timing(False)
gamma = 1.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=str, default="1,1,1")
    parser.add_argument("--num_segments", type=int, default=4)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument('--free', action='store_true')
    parser.add_argument('--pool_size', type=int, default=7, help="the pooled image size for next path initialization")
    parser.add_argument('--save_loss', action='store_true')
    return parser.parse_args()





def init_new_paths(num_paths, canvas_width, canvas_height, args, num_old_shapes=0, pixel_loss=None):
    shapes = []
    shape_groups = []

    # change path init location
    if pixel_loss is not None:
        region_loss = adaptive_avg_pool2d(pixel_loss. args.pool_size)
        sorted, indices = torch.sort(region_loss.reshape(-1), dim=0, descending=True)
        indices = indices[:num_paths]
        indices_h = torch.div(indices, args.pool_size, rounding_mode='trunc')
        indices_w = indices%(args.pool_size)
        norm_postion = torch.cat([indices_h.unsqueeze(dim=-1), indices_w.unsqueeze(dim=-1)], dim=-1)
        norm_postion = (norm_postion+0.5)/(args.pool_size + 1e-8)
        # print(f"norm_position equals: {norm_postion}")


    for i in range(num_paths):
        num_segments = args.num_segments
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.05
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            if j < num_segments - 1:
                points.append(p3)
                p0 = p3
        points = torch.tensor(points)
        if pixel_loss is not None:
            points = points-points.mean(dim=0, keepdim=True) + (norm_postion[i]).to(points.device)
        # print(f"new path shape is {points.shape}, max val: {torch.max(points)}, min val: {torch.min(points)}")
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points = num_control_points,
                             points = points,
                             stroke_width = torch.tensor(1.0),
                             is_closed = True)
        shapes.append(path)
        # !!!!!!problem is here. the shape group shape_ids is wrong
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([num_old_shapes+i]),
                                         fill_color = torch.tensor([random.random(),
                                                                    random.random(),
                                                                    random.random(),
                                                                    random.random()]))
        shape_groups.append(path_group)

    points_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)
    return shapes, shape_groups, points_vars, color_vars


def main():
    args = parse_args()
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    random.seed(1234)
    torch.manual_seed(1234)
    render = pydiffvg.RenderFunction.apply


    print(f"=> check init path ...")
    # initialize new shapes related stuffs.

    pixel_loss = torch.rand([1,1,240,240])
    pixel_loss[0,0,2,3]=100
    pixel_loss[0,0,120,120]=100
    pixel_loss[0,0,239,239]=100
    shapes, shape_groups, points_vars, color_vars = init_new_paths(
        3, 240, 240, args, 0, pixel_loss)
    save_name = 'check.svg'
    pydiffvg.save_svg(save_name, 240, 240, shapes, shape_groups)

    new_shapes = []
    for path in shapes:
        print(f"path point: {path.points}")
        print(f"path point shape is: {path.points.shape}")
        path.points = path.points-path.points.mean(dim=0,keepdim=True)+120
        new_shapes.append(path)
        # points_vars.append(path.points)
    pydiffvg.save_svg("check2.svg", 240, 240, new_shapes, shape_groups)
    # Optimize
    points_vars = [*points_vars]
    color_vars = [*color_vars]
    # print(f"control points are: {points_vars}")
    print(f"\nDone!\n")


if __name__ == "__main__":
    main()
