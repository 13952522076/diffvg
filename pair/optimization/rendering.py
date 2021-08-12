"""
python rendering.py --num_paths 256 --use_blob
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import argparse
import torch.nn.functional as F
import math
import os

pydiffvg.set_print_timing(False)
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())
render = pydiffvg.RenderFunction.apply
gamma = 1.0

def load_img(args):
    # filename = os.path.basename(args.target).split('.')[0]
    # target = torch.from_numpy(skimage.io.imread('imgs/lena.png')).to(torch.float32) / 255.0
    img_data = skimage.io.imread(args.target)
    if img_data.shape[2] == 4:
        print("Input image includes alpha channel, simply dropout alpha channel.")
        img_data = img_data[:, :, :3]
    target = torch.from_numpy(img_data).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2)  # NHWC -> NCHW
    return target

def main(args):
    # loading image.
    target = load_img(args)
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths = args.num_paths

    shapes = []
    shape_groups = []

    points_vars = []
    color_vars = []

    for i in range(num_paths):
        num_segments = args.num_segments
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []

        # change the detailed initlization to  will increase the loss from 0.0008 to 0.0017
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
        # print(f"points value is: {points}")
        # points = torch.rand([num_segments*3, 2])  # instead from p0=   to points = torch.tensor(points)

        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        points = torch.nn.Parameter(points)
        points_vars.append(points)
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(1.0),
                             is_closed=True)
        shapes.append(path)
        colors = torch.nn.Parameter(torch.rand(4))
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                         fill_color=colors)
        color_vars.append(colors)
        shape_groups.append(path_group)




    # for path in shapes:
    #     path.points.requires_grad = True
    #     points_vars.append(path.points)
    # for group in shape_groups:
    #     group.fill_color.requires_grad = True
    #     color_vars.append(group.fill_color)


    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    # color_optim = torch.optim.Adam(color_vars, lr=0.01)
    # points_optim = torch.optim.SGD(points_vars, lr=1.0)
    color_optim = torch.optim.SGD(color_vars, lr=0.01)
    # Adam iterations.
    for t in range(args.num_iter):
        points_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width,  # width
                     canvas_height,  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     t,  # seed
                     None,
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        # pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/{}_iter_{}.png'.format(filename,t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # loss = (img - target).pow(2).mean()
        loss = F.mse_loss(img, target)

        print(f'iteration: {t} \t render loss: {loss.item()}')

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        if args.use_blob:
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)
        else:
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

        # if t % 10 == 0 or t == args.num_iter - 1:
        if t == args.num_iter - 1:
            use_blob = "closed" if args.use_blob else "open"
            pydiffvg.save_svg('results/{}_iter_{}.svg'.format(use_blob,t),
                              canvas_width, canvas_height, shapes, shape_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  default='single.png', help="target image path")
    parser.add_argument("--num_paths", type=int, default=256)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=300)
    parser.add_argument("--num_segments", type=int, default=3)
    parser.add_argument("--folder", type=str, default="rendering")
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    args = parser.parse_args()
    main(args)
