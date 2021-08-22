"""
Based on ResMLP, add local conv in the MLP block.
"""
import math

import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Module):

    def __init__(self, dim, num_patch, mlp_dim, dropout = 0., init_values=1e-4):
        super().__init__()

        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patch, num_patch),
            Rearrange('b d n -> b n d'),
        )

        self.mid_affine = Aff(dim)
        self.local_mix = nn.Sequential(
            Rearrange('b (w h) d -> b d w h', w = int(math.sqrt(num_patch))),
            nn.Conv2d(dim, mlp_dim, 1, padding=0, stride=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim, dim, 3, padding=1, stride=1, groups=dim),
            Rearrange('b d w h -> b (w h) d'),
        )

        self.post_affine = Aff(dim)
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )

        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.mid_affine(x)
        x = x + self.gamma_2 * self.local_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_3 * self.ff(x)
        return x


class ConvResMLP3(nn.Module):

    def __init__(self, in_channels=3, dim=384, num_classes=384, patch_size=16, image_size=224, depth=12, mlp_dim=384*4):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, dim, patch_size, patch_size),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))

        self.affine = Aff(dim)

        # self.fc = nn.Sequential(
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, x):

        x = self.to_patch_embedding(x)

        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        x = self.affine(x)

        # x = x.mean(dim=1)

        return x




if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = ConvResMLP3()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
