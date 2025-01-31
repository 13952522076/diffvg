import torch
from base import BaseVAE
from torch import nn
from typing import List, TypeVar
from torchvision.models import resnet18
import torch.nn.functional as F

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

__all__ = ['VanillaVAE', "VAELoss"]


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        # Build Encoder
        modules.append(
            nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=64*4, kernel_size=4, stride=4),
                    nn.BatchNorm2d(64*4),
                    nn.LeakyReLU(),
                    nn.Conv2d(64*4, out_channels=hidden_dims[0], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU()
                )
        )
        in_channels=hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Conv2d(h_dim, out_channels=h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)


        # build mean and var_log
        self.fc_mu = nn.Linear(in_channels*4, latent_dim)
        self.fc_var = nn.Linear(in_channels*4, latent_dim)


        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], 4*hidden_dims[i + 1], kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_dims[-1], hidden_dims[-1]*4, kernel_size=3, stride=1, padding=1),
                            nn.PixelShuffle(2),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3*4*4, kernel_size= 3, padding= 1),
                            nn.PixelShuffle(4),
                            nn.Sigmoid())


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs: Tensor, **kwargs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return {
            "reconstruct": self.decode(z),
            "input": inputs,
            "mu": mu,
            "log_var": log_var
        }

    def sample(self, z, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        # z = torch.randn(num_samples,
        #                 self.latent_dim)
        #
        # z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)["reconstruct"]



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
        recons_loss = 100.0 * F.smooth_l1_loss(recons, input)

        kld_loss = kld_weight * torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}


if __name__ == '__main__':
    model = VanillaVAE(in_channels=3, latent_dim=256)
    x = torch.rand([3,3,128,128])
    out = model(x)
    reconstruct = out["reconstruct"]
    input = out["input"]
    mu = out["mu"]
    log_var = out["log_var"]
    print(reconstruct.shape)
    print(input.shape)
    print(mu.shape)
    print(log_var.shape)
