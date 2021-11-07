import torch
from base import BaseVAE
from torch import nn
from typing import List, TypeVar
from torchvision.models import resnet34
import torch.nn.functional as F

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

__all__ = ['VanillaAE', "VAELoss"]


class VanillaAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        resnet = resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        modules.append(nn.Flatten(start_dim=1))
        modules.append(nn.Linear(resnet.fc.in_features, 1024))
        modules.append(nn.BatchNorm1d(1024, momentum=0.01))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(1024, 1024))
        modules.append(nn.BatchNorm1d(1024, momentum=0.01))
        modules.append(nn.ReLU(inplace=True))
        resnet = nn.Sequential(*modules)


        # self.encoder = nn.Sequential(*modules)
        self.encoder = resnet

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(1024, 64*49)
        modules.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2), # 14
                    nn.Conv2d(64, 64, 3, padding="same"),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.UpsamplingBilinear2d(scale_factor=2), # 28
                    nn.Conv2d(64, 64, 3, padding="same"),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.UpsamplingBilinear2d(scale_factor=2), # 56
                    nn.Conv2d(64, 64, 3, padding="same"),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),

                    nn.UpsamplingBilinear2d(scale_factor=2), # 112
                    nn.Conv2d(64, 64, 3, padding="same"),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                    nn.Conv2d(64, 3*4, 3, padding="same"),
                    nn.PixelShuffle(2),
                    nn.Tanh()
                )
        )

        # hidden_dims.reverse()
        #
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.UpsamplingBilinear2d(scale_factor=2),
        #             nn.Conv2d(),
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride=2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU()
        #         )
        #     )

        self.decoder = nn.Sequential(*modules)



    def forward(self, inputs: Tensor, **kwargs):
        vector = self.encoder(inputs)
        out = self.decoder_input(vector)
        out = out.view(-1, 64, 7, 7)
        out = self.decoder(out)
        return out


class VAELoss(nn.Module):
    def __init__(self, M_N):
        super(VAELoss, self).__init__()
        self.M_N = M_N

    def forward(self, out):
        """
               Computes the VAE loss function.
               KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
               :param args:
               :param kwargs:
               :return:
               """
        recons = out["reconstruct"]
        input = out["input"]
        mu = out["mu"]
        log_var = out["log_var"]

        kld_weight = self.M_N  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = recons_loss
        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}


if __name__ == '__main__':
    model = VanillaAE(in_channels=3, latent_dim=512)
    x = torch.rand([3,3,224,224])
    out = model(x)
    reconstruct = out["reconstruct"]
    input = out["input"]
    mu = out["mu"]
    log_var = out["log_var"]
    print(reconstruct.shape)
    print(input.shape)
    print(mu.shape)
    print(log_var.shape)
