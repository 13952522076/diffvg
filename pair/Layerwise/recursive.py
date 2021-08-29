"""
python recursive.py ../data/emoji_rgb/validate/0/Emoji_u1f640.svg.png --num_paths 1,2
"""
import pydiffvg
import torch
import os
import skimage
import skimage.io
import random
import argparse
import math
import errno

pydiffvg.set_print_timing(False)
gamma = 1.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=str, default="1,2")
    parser.add_argument("--num_iter", type=int, default=500)
    return parser.parse_args()


try:
    os.makedirs("results/recursive")
except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir("results/recursive"):
            pass
        else:
            raise

def load_image(args):
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    print(f"Input image shape is: {target.shape}")
    if target.shape[2] == 4:
        print("Input image includes alpha channel, simply dropout alpha channel.")
        target = target[:, :, :3]
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0).permute(0, 3, 1, 2) # NHWC -> NCHW
    return target

def init_new_paths(num_paths, canvas_width, canvas_height):
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = 4
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
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points = num_control_points,
                             points = points,
                             stroke_width = torch.tensor(1.0),
                             is_closed = True)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
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
    filename = os.path.splitext(os.path.basename(args.target))[0]
    target = load_image(args)
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths_list = [int(i) for i in args.num_paths.split(',')]
    num_paths = num_paths_list[0]
    random.seed(1234)
    torch.manual_seed(1234)
    render = pydiffvg.RenderFunction.apply
    old_shapes, old_shape_groups = [], []




    # initialize new shapes related stuffs.
    shapes, shape_groups, points_vars, color_vars = init_new_paths(num_paths, canvas_width, canvas_height)
    if len(old_shapes)>0:
        for path in old_shapes:
            path.points.requires_grad = False
        for group in old_shape_groups:
            group.fill_color.requires_grad = False
    shapes = old_shapes+shapes
    shape_groups = old_shape_groups+shape_groups
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)
    # Adam iterations.
    for t in range(args.num_iter):
        points_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        if t == args.num_iter - 1:
            pydiffvg.imwrite(img.cpu(), 'results/recursive/{}_path_{}.png'.format(filename, args.num_paths), gamma=gamma)
        img = img[:, :, :3]
        img = img.unsqueeze(0).permute(0, 3, 1, 2) # HWC -> NCHW
        loss = (img - target).pow(2).mean()
        print(f'iteration: {t} \t render loss: {loss.item()}')
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)
        if t == args.num_iter - 1:
            pydiffvg.save_svg('results/recursive/{}_path_{}.svg'.format(filename, args.num_paths),
                              canvas_width, canvas_height, shapes, shape_groups)

if __name__ == "__main__":
    main()
