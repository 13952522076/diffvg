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
            nn.Linear(zdim, 2 * paths * (segments * 3 + 1)),
            nn.Tanh()
        )
        self.width_predictor = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, paths),  # width will be normalized to range [1, max]
            nn.Sigmoid()
        )
        self.color_predictor = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, paths * 4),  # color will be clamped to range [0,1]
            nn.Sigmoid()
        )
        # initialize parameters
        self._init_param()

    def _init_param(self):
        nn.init.constant_((self.width_predictor[2]).weight, 0)
        if (self.width_predictor[2]).bias is not None:
            torch.nn.init.constant_((self.width_predictor[2]).bias, 0)

    def forward(self, x):  # [b,z_dim]
        points = self.point_predictor(x)
        points = points * (self.im_size // 2) + self.im_size // 2
        widths = self.width_predictor(x)
        widths = (self.max_width - 1) * widths + 1
        colors = self.color_predictor(x)
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

class RealAE(nn.Module):
    def __init__(self, imsize=224, paths=512, segments=3, samples=2, zdim=2048,
                 max_width=2, pretained_encoder=True, **kwargs):
        super(RealAE, self).__init__()
        self.encoder = Encoder(zdim, pretrained=pretained_encoder)
        self.segments = segments
        self.paths = paths
        self.imsize = imsize
        self.samples = samples
        self.predictor = Predictor(zdim=zdim, paths=paths, segments=segments, max_width=max_width, im_size=imsize)
        # self.register_buffer("background",torch.ones(self.imsize, self.imsize, 3) * (1 - img[:, :, 3:4]))


    def get_batch_shapes_groups(self, predict_points, predict_widths, predict_colors):
        shapes_batch= []
        shape_groups_batch = []
        num_batch,num_paths, _, _ = predict_points.size()
        num_control_points = torch.zeros(self.segments, dtype=torch.int32) + 2
        for i in range(num_batch):
            shapes_image = []
            shape_groups_image = []
            for j in range(num_paths):
                path = pydiffvg.Path(num_control_points=num_control_points,
                                     points=predict_points[i,j,:,:],
                                     stroke_width=predict_widths[i,j],
                                     is_closed=True)
                shapes_image.append(path)

                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes_image) - 1]),
                                                 fill_color=predict_colors[i, j, :])
                shape_groups_image.append(path_group)
            shapes_batch.append(shapes_image)
            shape_groups_batch.append(shape_groups_image)
        return shapes_batch, shape_groups_batch

    def decoder(self, shapes_batch, shape_groups_batch):
        batch = len(shapes_batch)
        img_batch = []
        for i in range(batch):
            img = render(self.imsize, self.imsize, shapes_batch[i], shape_groups_batch[i], samples=self.samples)
            # Compose img with white background
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                              device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
            # Save the intermediate render.
            # pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/{}_iter_{}.png'.format(filename,t), gamma=gamma)
            img = img[:, :, :3]
            # Convert img from HWC to NCHW
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
            img_batch.append(img)

        return torch.cat(img_batch, dim=0)

    def forward(self, x):
        b, _, _, _ = x.size()
        z= self.encoder(x)
        predict = self.predictor(z)  # ["points" 2paths(3segments), "widths" paths, "colors" 4paths]
        predict_points = (predict["points"]).view(b, self.paths, -1, 2)
        predict_widths = (predict["widths"]).view(b, self.paths)
        predict_colors = (predict["colors"]).view(b, self.paths, 4)
        shapes_batch, shape_groups_batch = self.get_batch_shapes_groups(predict_points, predict_widths, predict_colors)
        out = self.decoder(shapes_batch, shape_groups_batch)

        return out

    def visualize(self, x, svgpath='demo.svg', inputpath='input.png', renderpath='render.png'):

        b, _, _, _ = x.size()
        if inputpath is not None:
            first_img = (x[0]).permute(1, 2, 0).cpu().numpy()
            plt.imsave(inputpath, first_img)
        z= self.encoder(x)
        predict = self.predictor(z)  # ["points" 2paths(3segments), "widths" paths, "colors" 4paths]
        predict_points = (predict["points"]).view(b, self.paths, self.segments*3, 2)
        predict_widths = (predict["widths"]).view(b, self.paths)
        predict_colors = (predict["colors"]).view(b, self.paths, 4)
        shapes_batch, shape_groups_batch = self.get_batch_shapes_groups(predict_points, predict_widths, predict_colors)
        shapes, shape_groups = shapes_batch[0], shape_groups_batch[0]

        scene_args = pydiffvg.RenderFunction.serialize_scene(self.imsize, self.imsize, shapes, shape_groups)
        img = _render(self.imsize, self.imsize, self.samples,  self.samples,   0,  None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        if renderpath is not None:
            pydiffvg.imwrite(img.cpu(), renderpath, gamma=1.0)
        if svgpath is not None:
            pydiffvg.save_svg(svgpath, self.imsize, self.imsize, shapes, shape_groups)


if __name__ == '__main__':
    encoder = Encoder(zdim=2048, pretrained=False)
    predictor = Predictor(zdim=2048, paths=512, segments=2, max_width=2.0, im_size=224.0)
    input = torch.rand([1, 3, 224, 224])
    embed_fea = encoder(input)
    predictions = predictor(embed_fea)
    print((predictions["points"]).shape)
    print((predictions["widths"]).shape)
    print((predictions["colors"]).shape)
    print((predictions["widths"]).shape)
    # print(predictor)

    # test  the pipeline
    img = torch.rand([2, 3, 224,224],device=pydiffvg.get_device())
    model = RealAE(imsize=224, paths=512, segments=3, samples=2, zdim=2048, max_width=2,
                 pretained_encoder=True)
    model.to("cuda")
    out = model(img)
    print(f"out shape is: {out.shape}")
    model.visualize(img, svgpath='demo.svg', inputpath='input.png', renderpath='render.png')
