"""
This is the main file or our proposed method.
Given an input image, we will progressively reconstruct it using svg Bezier path.

This will generate a folder named {args.save_folder}/{filename}/{details}

Here are some use cases:

python main_edge.py demo5.png --num_paths 1,1,1,1,1,1 --pool_size 40 --save_folder results/edge --free --num_segments 4


python main.py demo.png --num_paths 1,1,1,1,1,1 --pool_size 40 --save_folder video --free --save_video --num_segments 8

python main.py demo.png --num_paths 1,1,1,1,1,1 --pool_size 40 --save_folder circle --free --num_segments 4 --initial circle --circle_init_radius 0.01
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
import sys
sys.path.append("../Layerwise")
from XingLoss import xing_loss

pydiffvg.set_print_timing(False)
gamma = 1.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=str, default="1,1,1")
    parser.add_argument("--num_segments", type=int, default=4)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument('--free', action='store_true')
    # Please ensure that image resolution is divisible by pool_size; otherwise the performance would drop a lot.
    parser.add_argument('--pool_size', type=int, default=60, help="the pooled image size for next path initialization")
    parser.add_argument('--save_loss', action='store_true')
    parser.add_argument('--save_init', action='store_true')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--print_weight', action='store_true')
    parser.add_argument('--save_folder', metavar='DIR', default="output")
    parser.add_argument('--initial', type=str, default="random", choices=['random', 'circle'])
    parser.add_argument('--circle_init_radius',  type=float)
    parser.add_argument('--edge_weight',  type=float, default=0.1)
    parser.add_argument('--xing_weight',  type=float ,default=0.1, help="weight for crossing loss.")
    parser.add_argument('--threshold_max_path',  type=int ,default=30, help="weight for crossing loss.")
    parser.add_argument('--threshold_min_loss',  type=int ,default=0.0013,
                        help="0.0013 is decided by main.py demo.png, which fit an image very well")

    return parser.parse_args()


def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    m_point = (1.0, 0.0)
    points.append(m_point)
    m1_point_angle = np.degrees(4/3*np.tan(np.pi/(2*segments)))
    m1_point_radius = np.sqrt(1+ ( 4/3*np.tan(np.pi/(2*segments)))**2)
    m3_point_angle = 360/segments
    for i in range(0, segments):
        m1_point = (m1_point_radius*np.cos(np.deg2rad(i*m3_point_angle+m1_point_angle)),
                    m1_point_radius*np.sin(np.deg2rad(i*m3_point_angle+m1_point_angle)))
        m2_point_angle = m3_point_angle-m1_point_angle
        m2_point = (m1_point_radius*np.cos(np.deg2rad(i*m3_point_angle+m2_point_angle)),
                    m1_point_radius*np.sin(np.deg2rad(i*m3_point_angle+m2_point_angle)))
        m3_point = (np.cos(np.deg2rad(i*m3_point_angle+m3_point_angle)),
                    np.sin(np.deg2rad(i*m3_point_angle+m3_point_angle)))
        points.append(m1_point)
        points.append(m2_point)
        points.append(m3_point)
    points.reverse()
    points = torch.tensor(points)
    points = points[:-1,:]
    points = (points)*radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points


def make_save_path(args):
    filename = os.path.splitext(os.path.basename(args.target))[0]
    detail_folder = args.num_paths+"Seg"+str(args.num_segments)+"Iter"+str(args.num_iter)+"Pool"+str(args.pool_size)
    if args.free:
        detail_folder+="Free"
    detail_folder+=args.initial
    if args.initial=='circle' and args.circle_init_radius is not None:
        detail_folder+=str(args.circle_init_radius)
    detail_folder = detail_folder + "Edge" + str(args.edge_weight)
    save_path = os.path.join(args.save_folder, filename, detail_folder)
    try:
        os.makedirs(save_path)
    except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(save_path):
                pass
            else:
                raise
    if args.save_video:
        try:
            os.makedirs(os.path.join(save_path,'videos'))
            os.makedirs(os.path.join(save_path,'images'))
            os.makedirs(os.path.join(save_path,'edges'))
        except OSError as exc:  # Python >2.5
            pass
    return save_path


def load_image(args):
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    print(f"Input image shape is: {target.shape}")
    if target.shape[2] == 4:
        print("Input image includes alpha channel, simply dropout alpha channel.")
        target = target[:, :, :3]
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0).permute(0, 3, 1, 2) # NHWC -> NCHW

    # detection edge
    image = cv2.imread(args.target)# read image
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(image,30, 50)
    target_edge = torch.from_numpy(edge) /255.0
    target_edge = target_edge.to(pydiffvg.get_device())
    print(f"target_edge shape is: {target_edge.shape}")
    # unique have a look
    print(f"target_edge  max: {target_edge.max()} | min: {target_edge.min()}")
    return target, target_edge


def init_new_paths(num_paths, canvas_width, canvas_height, args, num_old_shapes=0, pixel_loss=None):
    shapes = []
    shape_groups = []

    # change path init location
    if pixel_loss is not None:

        region_loss = adaptive_avg_pool2d(pixel_loss, args.pool_size)
        sorted, indices = torch.sort(region_loss.reshape(-1), dim=0, descending=True)
        indices = indices[:num_paths]
        indices_h = torch.div(indices, args.pool_size, rounding_mode='trunc')
        indices_w = indices%(args.pool_size)
        # norm_postion = torch.cat([indices_h.unsqueeze(dim=-1), indices_w.unsqueeze(dim=-1)], dim=-1)
        # [w,h] for diffvg
        norm_postion = torch.cat([indices_w.unsqueeze(dim=-1), indices_h.unsqueeze(dim=-1)], dim=-1)
        norm_postion = (norm_postion+0.5)/(args.pool_size)
        # print(f"norm_position equals: {norm_postion}")


    for i in range(num_paths):
        num_segments = args.num_segments
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2

        #### original point initialization
        if args.initial=="random":
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

        # circle points initialization
        else:
            radius = args.circle_init_radius
            if radius is None:
                radius = np.random.uniform(low=0.003, high=0.03)
            print(f"radius {radius} for circle initialization")
            points = get_bezier_circle(radius=radius, segments=num_segments, bias=(random.random(), random.random()))



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
    print(f"points shape is: {(points_vars[0]).shape}")
    return shapes, shape_groups, points_vars, color_vars


def plot_loss_map(pixel_loss, args,savepath="./"):
    _, _, H, W = pixel_loss.size()
    region_loss = adaptive_avg_pool2d(pixel_loss, args.pool_size)
    region_loss = region_loss.squeeze(dim=0).squeeze(dim=0)
    region_loss = torch.softmax(region_loss.view(-1),dim=0).reshape(args.pool_size, args.pool_size).cpu().detach().numpy()
    pixel_loss = pixel_loss.squeeze(dim=0).squeeze(dim=0)
    pixel_loss = torch.softmax(pixel_loss.view(-1),dim=0).reshape(H,W).cpu().detach().numpy()
    plt.imshow(pixel_loss, cmap='Reds')
    plt.colorbar()
    plt.savefig(savepath+"-loss_pixel.png", dpi=800)
    plt.close()
    plt.imshow(region_loss, cmap='Reds')
    plt.colorbar()
    plt.savefig(savepath+"-loss_region.png", dpi=800)
    plt.close()

render = pydiffvg.RenderFunction.apply

def main_single_img():
    args = parse_args()
    save_path = make_save_path(args)
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    # filename = os.path.splitext(os.path.basename(args.target))[0]
    target, target_edge = load_image(args)
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    random.seed(1234)
    torch.manual_seed(1234)

    row_data_list = []

    old_shapes, old_shape_groups = [], []
    pixelwise_loss = target.mean(dim=2,keepdim=False)

    num_segments_options = [3,4,5,6,7,8]
    color_options = ["Normal", "RadialGradient", "LinearGradient"]

    for threshold_path in range(0, args.threshold_max_path):
        best_num_segments = None
        best_color = None
        best_loss = 9999.
        # initialize the row data
        row_data = {"pixelwise_loss": pixelwise_loss,
                    "best_num_segments": best_num_segments,
                    "best_color": best_color,
                    "best_loss": best_loss
                    }
        for num_segment in num_segments_options:
            for color_option in color_options:
                candidate_old_shapes, candidate_old_shape_groups, candidate_pixelwise_loss, candidate_loss  = detail_method(
                    old_shapes, old_shape_groups, pixelwise_loss, num_segment, color_option,
                    target, target_edge, canvas_width, canvas_height, args)
                if candidate_loss < best_loss:
                    best_old_shapes = candidate_old_shapes
                    best_old_shape_groups = candidate_old_shape_groups
                    pixelwise_loss = candidate_pixelwise_loss
                    best_loss = candidate_loss
                    best_num_segments = num_segment
                    best_color = color_option
        old_shapes, old_shape_groups = best_old_shapes, best_old_shape_groups

        # append the row data
        row_data.best_num_segments = best_num_segments
        row_data.best_color = best_color
        row_data.best_loss = best_loss
        row_data_list.append(row_data)

        if best_loss < args.threshold_min_loss:
            break

    print(f"\nDone! total {threshold_path+1} paths, the last loss is: {best_loss.item()}.\n")
    return row_data_list


def detail_method(old_shapes, old_shape_groups, pixelwise_loss, num_segment, color_option,
                    target, target_edge, canvas_width, canvas_height, args):
    """
    !!! The two gradient colors are not implemented yet!!!
    """
    # calculate regional loss_weight
    region_loss = adaptive_avg_pool2d(pixelwise_loss, args.pool_size)
    loss_weight = torch.softmax(region_loss.reshape(1,1,-1),dim=-1).reshape_as(region_loss)
    loss_weight = torch.nn.functional.interpolate(loss_weight, size=[canvas_height,canvas_width], mode='area')
    loss_weight = loss_weight/(loss_weight.sum())
    loss_weight = loss_weight.clone().detach()

    shapes, shape_groups, points_vars, color_vars = init_new_paths(
        1, canvas_width, canvas_height, args, len(old_shapes), pixelwise_loss)
    old_points_vars = []
    old_color_vars = []
    copyed_shapes = []
    copyed_shape_groups = []
    if len(old_shapes) > 0:
        for old_path in old_shapes:
            copyed_path = old_path.clone()
            copyed_shapes.append(copyed_path)
            if args.free:
                copyed_path.points.requires_grad = True
                old_points_vars.append(copyed_path.points)
            else:
                copyed_path.points.requires_grad = False
        for old_group in old_shape_groups:
            copyed_group = old_group.clone()
            copyed_shape_groups.append(copyed_group)
            if args.free:
                copyed_group.fill_color.requires_grad = True
                old_color_vars.append(copyed_group.fill_color)
            else:
                copyed_group.fill_color.requires_grad = False

    shapes = [*copyed_shapes, *shapes]
    shape_groups = [*copyed_shape_groups, *shape_groups]

    # Optimize
    points_vars = [*old_points_vars, *points_vars]
    color_vars = [*old_color_vars, *color_vars]
    points_optim = torch.optim.Adam(points_vars, lr=1)
    color_optim = torch.optim.Adam(color_vars, lr=0.1)
    points_scheduler = CosineAnnealingLR(points_optim, args.num_iter, eta_min=0.5)
    color_scheduler = CosineAnnealingLR(color_optim, args.num_iter, eta_min=0.05)
    # Adam iterations.
    t_range = tqdm(range(args.num_iter))
    for t in t_range:
        points_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])

        #### start save edge ###
        edge_groups = []
        for shape_group in shape_groups:
            edge_groups.append(
                pydiffvg.ShapeGroup(shape_ids = shape_group.shape_ids,
                                    fill_color = torch.tensor([0., 0., 0., 0.]),
                                    stroke_color = torch.tensor([1., 1., 1., 1.])
                                    )
            )
        edge_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, edge_groups)
        edge_img = render(canvas_width, canvas_height, 2, 2, t, None, *edge_args)
        edge_img = edge_img[:, :, 3:4] * edge_img[:, :, :3] + torch.zeros(edge_img.shape[0], edge_img.shape[1], 3, device = pydiffvg.get_device()) * (1 - edge_img[:, :, 3:4])
        # save_name = os.path.join(save_path,"edges", f"{current_path_str[:-1]}-{t}.png")
        # pydiffvg.imwrite(edge_img.cpu(), save_name, gamma=gamma)
        edge_img = edge_img[:, :, :3]
        # this can ensure value to be 0 or 1
        edge_img = 0.299*edge_img[:,:,0] + 0.587*edge_img[:,:,1] + 0.114*edge_img[:,:,2]
        #### end   save edge ###

        img = img[:, :, :3]
        img = img.unsqueeze(0).permute(0, 3, 1, 2) # HWC -> NCHW
        # loss = (img - target).pow(2).mean(dim=1,keepdim=True)
        pixelwise_loss = 0.
        mse_loss = ((img-target)**2).sum(dim=1, keepdim=True) # [N,1,H, W]
        pixelwise_loss += mse_loss
        # add edge loss here
        edge_loss = args.edge_weight * (abs(target_edge - edge_img)).unsqueeze(dim=0).unsqueeze(dim=0)  # [1,1,H,W]
        pixelwise_loss += edge_loss
        # add xing_loss here
        x_loss = xing_loss(points_vars,scale=args.xing_weight)  # real value [1]
        # pixelwise_loss += x_loss  # pixel-wise loss should not consider the x_los in cal since it is a real value.
        t_range.set_postfix({'mse_loss': mse_loss.mean().item(), 'edge_loss': edge_loss.mean().item(),'xing_loss': x_loss.item()})
        # Backpropagate the gradients.
        loss = ((mse_loss + edge_loss + x_loss)*loss_weight).sum()
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        points_scheduler.step()
        color_scheduler.step()

        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

    old_shapes = shapes
    old_shape_groups = shape_groups

    # calculate the pixel loss
    pixel_loss = ((img-target)**2).sum(dim=1, keepdim=True).sqrt_() # [N,1,H, W]
    edge_loss = args.edge_weight*((target_edge-edge_img)**2).unsqueeze(dim=0).unsqueeze(dim=0).sqrt_() # [N,1,H, W]
    region_loss = adaptive_avg_pool2d(pixel_loss+edge_loss, args.pool_size)
    loss_weight = torch.softmax(region_loss.reshape(1,1,-1),dim=-1).reshape_as(region_loss)
    loss_weight = torch.nn.functional.interpolate(loss_weight, size=[canvas_height,canvas_width], mode='area')
    loss_weight = loss_weight/(loss_weight.sum())
    loss_weight = loss_weight.clone().detach()


if __name__ == "__main__":
    main_single_img()
