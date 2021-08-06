#!/bin/env python
"""Train a AE Fashion-MNIST.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydiffvg
from torchvision.models import resnet50
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pydiffvg.set_use_gpu(torch.cuda.is_available())


class Predictor(nn.Module):
    def __init__(self, paths=512, segments=2, max_width=2.0, im_size=224.0):
        super(Predictor, self).__init__()
        self.points = nn.Parameter(2 * paths * (segments * 3 + 1))
        self.widths = nn.Parameter(paths)
        self.colors = nn.Parameter(paths*4)
        self.max_width = max_width
        self.im_size = im_size

    def forward(self):  # [b,z_dim]
        points = F.tanh(self.points)
        points = points * (self.im_size // 2) + self.im_size // 2
        widths = F.sigmoid(self.widths)
        widths = (self.max_width - 1) * widths + 1
        colors = F.sigmoid(self.colors)
        return {
            "points": points,
            "widths": widths,
            "colors": colors
        }


_render = pydiffvg.RenderFunction.apply


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,
                  canvas_height,
                  samples,
                  samples,
                  0,
                  None,
                  *scene_args)
    return img

class RealOptimize(nn.Module):
    def __init__(self, imsize=224, paths=512, segments=3, samples=2, zdim=2048,
                 max_width=2, **kwargs):
        super(RealOptimize, self).__init__()
        self.segments = segments
        self.paths = paths
        self.imsize = imsize
        self.samples = samples
        self.predictor = Predictor(paths=paths, segments=segments, max_width=max_width, im_size=imsize)
        # self.register_buffer("background",torch.ones(self.imsize, self.imsize, 3) * (1 - img[:, :, 3:4]))

    def get_shapes_groups(self, predict_points, predict_widths, predict_colors):
        num_paths, _, _ = predict_points.size()
        num_control_points = torch.zeros(self.segments, dtype=torch.int32) + 2
        shapes_image = []
        shape_groups_image = []
        for j in range(num_paths):
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=predict_points[j,:,:],
                                 stroke_width=predict_widths[j],
                                 is_closed=True)
            shapes_image.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes_image) - 1]),
                                             fill_color=predict_colors[j, :])
            shape_groups_image.append(path_group)
        return shapes_image, shape_groups_image

    def decoder(self, shapes, shape_groups):
        img = render(self.imsize, self.imsize, shapes, shape_groups, samples=self.samples)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        # pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/{}_iter_{}.png'.format(filename,t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img

    def forward(self):
        predict = self.predictor()  # ["points" 2paths(3segments), "widths" paths, "colors" 4paths]
        predict_points = (predict["points"]).view(self.paths, -1, 2)
        predict_widths = (predict["widths"]).view(self.paths)
        predict_colors = (predict["colors"]).view(self.paths, 4)
        shapes, shape_groups = self.get_shapes_groups(predict_points, predict_widths, predict_colors)
        out = self.decoder(shapes, shape_groups)
        return out

    def visualize(self, svgpath='demo.svg', inputpath='input.png', renderpath='render.png'):
        predict = self.predictor()  # ["points" 2paths(3segments), "widths" paths, "colors" 4paths]
        predict_points = (predict["points"]).view(self.paths, -1, 2)
        predict_widths = (predict["widths"]).view(self.paths)
        predict_colors = (predict["colors"]).view(self.paths, 4)
        shapes, shape_groups = self.get_shapes_groups(predict_points, predict_widths, predict_colors)
        scene_args = pydiffvg.RenderFunction.serialize_scene(self.imsize, self.imsize, shapes, shape_groups)
        img = _render(self.imsize, self.imsize, self.samples,  self.samples,   0,  None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        if renderpath is not None:
            pydiffvg.imwrite(img.cpu(), renderpath, gamma=1.0)
        if svgpath is not None:
            pydiffvg.save_svg(svgpath, self.imsize, self.imsize, shapes, shape_groups)


if __name__ == '__main__':

    model = RealOptimize(imsize=224, paths=512, segments=3, samples=2, zdim=2048, max_width=2)
    model.to("cuda")
    out = model()
    print(f"out shape is: {out.shape}")
    model.visualize(svgpath='demo.svg', inputpath='input.png', renderpath='render.png')
