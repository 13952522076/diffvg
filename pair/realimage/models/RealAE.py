#!/bin/env python
"""Train a AE Fashion-MNIST.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydiffvg
from torchvision.models import resnet50


class Encoder(nn.Module):
    def __init__(self, zdim=2048, pretrained=False):
        super(Encoder, self).__init__()
        net = resnet50(pretrained=pretrained)
        net.fc = nn.Linear(2048, zdim)
        self.net = net

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, zdim=2048, paths=512, segments=2, max_width=2.0, im_size=224.0):
        super(Predictor, self).__init__()
        self.max_width = max_width
        self.im_size = im_size
        # self.num_control_points = torch.zeros(segments, dtype=torch.int32) + 2
        self.point_predictor = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, 2 * paths * (segments * 3 + 1))
        )
        self.width_predictor = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, paths)  # width will be normalized to range [1, max]
        )
        self.color_predictor = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, paths * 4)  # color will be clamped to range [0,1]
        )
        # initialize parameters
        self._init_param()

    def _init_param(self):
        nn.init.constant_((self.width_predictor[2]).weight, 0)
        if (self.width_predictor[2]).bias is not None:
            torch.nn.init.constant_((self.width_predictor[2]).bias, 0)

    def forward(self, x):  # [b,z_dim]
        points = self.point_predictor(x)
        points *= self.im_size
        widths = self.width_predictor(x)
        widths = torch.clamp(widths, 1.0, self.max_width)
        colors = self.color_predictor(x)
        colors = torch.clamp(colors, 0, 1)
        return {
            "points": points,
            "widths": widths,
            "colors": colors
        }


def render(canvas_width, canvas_height, shapes, shape_groups, samples=2):
    _render = pydiffvg.RenderFunction.apply
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

class RealAE(nn.Module):
    def __init__(self, imsize=224, paths=512, segments=3, samples=2, zdim=2048, max_width=2,
                 pretained_encoder=True, **kwargs):
        super(RealAE, self).__init__()
        self.encoder = Encoder(zdim, pretrained=pretained_encoder)
        self.segments = segments
        self.paths = paths
        self.imsize = imsize
        self.samples = samples
        self.predictor = Predictor(zdim=zdim, paths=paths, segments=segments, max_width=max_width, im_size=imsize)

    def decoder(self, z):
        bs = z.shape[0]
        # predict points, points are centeralized
        all_points = self.point_predictor(z).view(bs, self.paths, -1, 2)
        all_points = all_points * (self.imsize // 2 - 2) + self.imsize // 2

        # predict width, min-max normalization to range [min-max]
        all_widths = self.width_predictor(z)
        min_width = self.stroke_width[0]
        max_width = self.stroke_width[1]
        all_widths = (max_width - min_width) * all_widths + min_width

        # predict alpha channel
        all_alphas = self.alpha_predictor(z)

        # Process the batch sequentially
        outputs = []
        scenes = []
        for k in range(bs):
            # Get point parameters from network
            shapes = []
            shape_groups = []
            for p in range(self.paths):
                points = all_points[k, p].contiguous().cpu()
                width = all_widths[k, p].cpu()
                alpha = all_alphas[k, p].cpu()
                color = torch.cat([torch.ones(3), alpha.view(1, )])
                num_ctrl_pts = torch.zeros(self.segments, dtype=torch.int32) + 2
                path = pydiffvg.Path(num_control_points=num_ctrl_pts, points=points, stroke_width=width, is_closed=False)
                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=None,stroke_color=color)
                shape_groups.append(path_group)
            scenes.append([shapes, shape_groups, (self.imsize, self.imsize)])
            # Rasterize
            out = render(self.imsize, self.imsize, shapes, shape_groups, samples=self.samples)
            # Torch format, discard alpha, make gray
            out = out.permute(2, 0, 1).view(4, self.imsize, self.imsize)[:3].mean(0, keepdim=True)
            outputs.append(out)
        output = torch.stack(outputs).to(z.device)
        return output


    def forward(self, x):
        z= self.encoder(x)
        predict = self.predictor(z)  # ["points" 2paths(3segments+1), "widths" paths, "colors" 4paths]
        output = self.decoder(z)
        return output


if __name__ == '__main__':
    encoder = Encoder(zdim=2048, pretrained=False)
    predictor = Predictor(zdim=2048, paths=512, segments=2, max_width=2.0, im_size=224.0)
    input = torch.rand([1, 3, 224, 224])
    embed_fea = encoder(input)
    predictions = predictor(embed_fea)
    print((predictions["points"]).shape)
    print((predictions["widths"]).shape)
    print((predictions["colors"]).shape)
    print((predictions["widths"]))
    # print(predictor)
