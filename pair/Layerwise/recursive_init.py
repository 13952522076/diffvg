"""
python recursive_init.py demo.png --num_paths 1,1,1,1 --save_loss
"""
import pydiffvg
import torch
import os
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


try:
    os.makedirs("results/recursive_init")
except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir("results/recursive_init"):
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
        norm_postion = torch.cat([indices_h.unsqueeze(dim=-1), indices_w.unsqueeze(dim=-1)], dim=-1)
        norm_postion = (norm_postion+0.5)/(args.pool_size)
        # print(f"norm_position equals: {norm_postion}")


    for i in range(num_paths):
        num_segments = args.num_segments
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        # if pixel_loss is not None:
        #     p0 = (norm_postion[i]).to(points.device)
        # else:
        #     p0 = (random.random(), random.random())
        p0 = (random.random(), random.random())
        # p0 = norm_postion[i]
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


def plot_loss_map(pixel_loss, args,savepath="./"):
    _, _, H, W = pixel_loss.size()
    region_loss = adaptive_avg_pool2d(pixel_loss, args.pool_size)
    region_loss = region_loss.squeeze(dim=0).squeeze(dim=0)
    region_loss = torch.softmax(region_loss.view(-1),dim=0).reshape(args.pool_size, args.pool_size).cpu().detach().numpy()
    pixel_loss = pixel_loss.squeeze(dim=0).squeeze(dim=0)
    pixel_loss = torch.softmax(pixel_loss.view(-1),dim=0).reshape(H,W).cpu().detach().numpy()
    plt.imshow(pixel_loss, cmap='Reds')
    plt.colorbar()
    plt.savefig(savepath+"-loss_pixel.pdf", dpi=800)
    plt.close()
    plt.imshow(region_loss, cmap='Reds')
    plt.colorbar()
    plt.savefig(savepath+"-loss_region.pdf", dpi=800)
    plt.close()



def main():
    args = parse_args()
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    filename = os.path.splitext(os.path.basename(args.target))[0]
    target = load_image(args)
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths_list = [int(i) for i in args.num_paths.split(',')]
    random.seed(1234)
    torch.manual_seed(1234)
    render = pydiffvg.RenderFunction.apply

    current_path_str = ""
    old_shapes, old_shape_groups = [], []

    region_loss = None
    loss_weight = 1.0/(canvas_width*canvas_height)
    for num_paths in num_paths_list:
        print(f"\n=> Adding {num_paths} paths ...")
        current_path_str = current_path_str+str(num_paths)+","
        # initialize new shapes related stuffs.
        shapes, shape_groups, points_vars, color_vars = init_new_paths(
            num_paths, canvas_width, canvas_height, args, len(old_shapes), region_loss)
        old_points_vars = []
        old_color_vars = []
        if len(old_shapes)>0:
            for old_path in old_shapes:
                if args.free:
                    old_path.points.requires_grad = True
                    old_points_vars.append(old_path.points)
                else:
                    old_path.points.requires_grad = False
            for old_group in old_shape_groups:
                if args.free:
                    old_group.fill_color.requires_grad = True
                    old_color_vars.append(old_group.fill_color)

                else:
                    old_group.fill_color.requires_grad = False
        # if len(old_shapes) >0:
        #     print(f"old shapes first path points is: {(old_shapes[0]).points}")
        #     print(f"new shapes first path points is: {(shapes[0]).points}")
        shapes = [*old_shapes, *shapes]
        shape_groups = [*old_shape_groups, *shape_groups]
        save_name = 'results/recursive_init/{}_path{}[{}]-segments{}'.\
            format(filename, args.num_paths,current_path_str[:-1], args.num_segments)
        if args.free:
            save_name+='-free'
        save_name+='-init.svg'
        # for shap in shapes:
        #     print(shap.points)
        pydiffvg.save_svg(save_name, canvas_width, canvas_height, shapes, shape_groups)
        # Optimize
        points_vars = [*old_points_vars, *points_vars]
        color_vars = [*old_color_vars, *color_vars]
        points_optim = torch.optim.Adam(points_vars, lr=1.0)
        color_optim = torch.optim.Adam(color_vars, lr=0.01)
        points_scheduler = CosineAnnealingLR(points_optim, args.num_iter, eta_min=0.1)
        color_scheduler = CosineAnnealingLR(color_optim, args.num_iter, eta_min=0.001)
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
            # if t == 0:
            #     save_name = 'results/recursive_init/{}_path{}[{}]-segments{}'.\
            #         format(filename, args.num_paths,current_path_str[:-1], args.num_segments)
            #     if args.free:
            #         save_name+='-free'
            #     save_name+='.svg'
            #     pydiffvg.save_svg(save_name, canvas_width, canvas_height, shapes, shape_groups)
            #     pydiffvg.imwrite(img.cpu(), 'results/recursive/{}_path{}[{}].png'.
            #                      format(filename, args.num_paths, current_path_str[:-1]), gamma=gamma)
            img = img[:, :, :3]
            img = img.unsqueeze(0).permute(0, 3, 1, 2) # HWC -> NCHW
            # loss = (img - target).pow(2).mean(dim=1,keepdim=True)
            loss = ((img-target)**2).sum(dim=1, keepdim=True).sqrt_() # [N,1,H, W]
            loss = (loss*loss_weight).sum()
            # print(f'iteration: {t} \t render loss: {loss.item()}')
            t_range.set_postfix({'loss': loss.item()})
            # Backpropagate the gradients.
            loss.backward()

            # Take a gradient descent step.
            points_optim.step()
            color_optim.step()

            points_scheduler.step()
            color_scheduler.step()

            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)
            if t == args.num_iter - 1:
                save_name = 'results/recursive_init/{}_path{}[{}]-segments{}'.\
                    format(filename, args.num_paths,current_path_str[:-1], args.num_segments)
                if args.free:
                    save_name+='-free'
                save_name+=f'-i{t}.svg'
                pydiffvg.save_svg(save_name, canvas_width, canvas_height, shapes, shape_groups)



        old_shapes = shapes
        old_shape_groups = shape_groups

        # calculate the pixel loss
        pixel_loss = ((img-target)**2).sum(dim=1, keepdim=True).sqrt_() # [N,1,H, W]
        region_loss = adaptive_avg_pool2d(pixel_loss, args.pool_size)
        # region_loss = region_loss - region_loss.mean()
        loss_weight = torch.softmax(region_loss.reshape(1,1,-1),dim=-1).reshape_as(region_loss)
        # loss_weight = region_loss/region_loss.sum()
        print(f"softmax loss weight is: \n{loss_weight}")
        loss_weight = torch.nn.functional.interpolate(loss_weight, size=[canvas_height,canvas_width], mode='area')
        loss_weight = loss_weight/(loss_weight.sum())
        # print(f"loss_weight shape is {loss_weight.shape}")
        # print(f"loss_weight is {loss_weight}")
        loss_weight = loss_weight.clone().detach()
        if args.save_loss:
            print("start saving loss heatmap...")
            save_name = 'results/recursive_init/{}_path{}[{}]-segments{}'.\
                    format(filename, args.num_paths,current_path_str[:-1], args.num_segments)
            if args.free:
                save_name+='-free'
            save_name+=f'-i{t}'
            plot_loss_map(pixel_loss, args, savepath=save_name)
            print("end saving loss heatmap...")

        # print(f"Top {num_paths} losses are {norm_postion}")




    print(f"\nDone! total {sum(num_paths_list)} paths, the last loss is: {loss.item()}.\n")


if __name__ == "__main__":
    main()
