import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, zdim=2048, paths=512, segments=2, im_size=224.0):
        super(Predictor, self).__init__()
        self.im_size = im_size
        # self.num_control_points = torch.zeros(segments, dtype=torch.int32) + 2
        self.point_predictor = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, 2 * paths * (segments * 3 + 1)),
            nn.Tanh()
        )
        self.color_predictor = nn.Sequential(
            nn.Linear(zdim, zdim),
            nn.ReLU(inplace=True),
            nn.Linear(zdim, paths * 4),  # color will be clamped to range [0,1]
            nn.Sigmoid()
        )


    def forward(self, x):  # [b,z_dim]
        points = self.point_predictor(x)
        points = points * (self.im_size // 2) + self.im_size // 2
        colors = self.color_predictor(x)
        return {
            "points": points,
            "colors": colors
        }


net = Predictor()
print(net.point_predictor[2])
