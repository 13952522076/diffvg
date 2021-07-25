#!/bin/env python
"""Train a AE Fashion-MNIST.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pydiffvg
from torchvision.models import resnet50

class Encoder(nn.Module):
 def __init__(self, zdim=1024, pretrained=True):
     net = resnet50(pretrained=pretrained)
     #net.fc = nn.Linear(2048,zdim)
     print(net.fc)


# def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
#     _render = pydiffvg.RenderFunction.apply
#     scene_args = pydiffvg.RenderFunction.serialize_scene(
#         canvas_width, canvas_height, shapes, shape_groups)
#     img = _render(canvas_width,
#                   canvas_height,
#                   samples,
#                   samples,
#                   0,
#                   None,
#                   *scene_args)
#     return img
#
#
# class Encoder(nn.Module):
#     def __init__(self, zdim=256):
#         super(Encoder, self).__init__()
#         self.encoder = torch.nn.Sequential(
#             # 28*28
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # 14*14
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             # 7*7
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             # 5*5
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             # 3*3
#             nn.Flatten(),
#             nn.Linear(in_features=128 * 3 * 3, out_features=zdim),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self,x):
#         return self.encoder(x)
#
#
# class VectorMNISTAE(nn.Module):
#     def __init__(self, imsize=28, paths=4, segments=5, samples=2, zdim=1024, stroke_width=None, **kwargs):
#         super(VectorMNISTAE, self).__init__()
#         self.encoder = Encoder(zdim)
#         self.segments = segments
#         self.paths = paths
#         self.imsize = imsize
#         self.samples = samples
#         self.stroke_width = stroke_width
#         self.stroke_width = (0.0, 2.0) if stroke_width is None else stroke_width
#
#         # 4 points bezier with n_segments -> 3*n_segments + 1 points
#         self.point_predictor = nn.Sequential(
#             nn.Linear(zdim, zdim),
#             nn.ReLU(inplace=True),
#             nn.Linear(zdim, 2 * self.paths * (self.segments * 3 + 1)),
#             nn.Tanh()  # bound spatial extent [-1,1]
#         )
#
#         self.width_predictor = nn.Sequential(
#             nn.Linear(zdim, zdim),
#             nn.ReLU(inplace=True),
#             nn.Linear(zdim, self.paths),
#             nn.Sigmoid()
#         )
#
#         self.alpha_predictor = nn.Sequential(
#             nn.Linear(zdim, zdim),
#             nn.ReLU(inplace=True),
#             nn.Linear(zdim, self.paths),
#             nn.Sigmoid()
#         )
#
#     def decoder(self, z):
#         bs = z.shape[0]
#         # predict points, points are centeralized
#         all_points = self.point_predictor(z).view(bs, self.paths, -1, 2)
#         all_points = all_points * (self.imsize // 2 - 2) + self.imsize // 2
#
#         # predict width, min-max normalization to range [min-max]
#         all_widths = self.width_predictor(z)
#         min_width = self.stroke_width[0]
#         max_width = self.stroke_width[1]
#         all_widths = (max_width - min_width) * all_widths + min_width
#
#         # predict alpha channel
#         all_alphas = self.alpha_predictor(z)
#
#         # Process the batch sequentially
#         outputs = []
#         scenes = []
#         for k in range(bs):
#             # Get point parameters from network
#             shapes = []
#             shape_groups = []
#             for p in range(self.paths):
#                 points = all_points[k, p].contiguous().cpu()
#                 width = all_widths[k, p].cpu()
#                 alpha = all_alphas[k, p].cpu()
#                 color = torch.cat([torch.ones(3), alpha.view(1, )])
#                 num_ctrl_pts = torch.zeros(self.segments, dtype=torch.int32) + 2
#                 path = pydiffvg.Path(num_control_points=num_ctrl_pts, points=points, stroke_width=width, is_closed=False)
#                 shapes.append(path)
#                 path_group = pydiffvg.ShapeGroup(
#                     shape_ids=torch.tensor([len(shapes) - 1]),
#                     fill_color=None,stroke_color=color)
#                 shape_groups.append(path_group)
#             scenes.append([shapes, shape_groups, (self.imsize, self.imsize)])
#             # Rasterize
#             out = render(self.imsize, self.imsize, shapes, shape_groups, samples=self.samples)
#             # Torch format, discard alpha, make gray
#             out = out.permute(2, 0, 1).view(4, self.imsize, self.imsize)[:3].mean(0, keepdim=True)
#             outputs.append(out)
#         output = torch.stack(outputs).to(z.device)
#         return output
#
#
#     def forward(self, x):
#         z= self.encoder(x)
#         output = self.decoder(z)
#         return output
#

if __name__ == '__main__':
    encoder = Encoder()

