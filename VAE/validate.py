from __future__ import print_function

import torch
import torch.optim as optim
import torchvision
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import argparse
from utils import mkdir_p, Logger, progress_bar, save_model, save_binary_img
from torchvision.datasets import ImageFolder
from vae import VAELoss,VanillaVAE


parser = argparse.ArgumentParser(description='VAE training for NSF project')

# General MODEL parameters
parser.add_argument('--latent_dim', default=256, type=int)
# Parameters for  dataset
parser.add_argument('--input', default='../pair/data/face/0/240px-Emoji_u1f61c.svg copy.png', type=str)
parser.add_argument('--testdir', default='../pair/data/emoji_rgb/validate')

# Parameters for  training
parser.add_argument('--resume', default='./checkpoints/VanillaVAE-256/checkpoint_best.pth', type=str, metavar='PATH', help='path to latest checkpoint')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args.checkpoint = './checkpoints/%s-%s' % ( "VanillaVAE", args.latent_dim)

print('==> Load image..')
image = Image.open(args.input)
x = TF.to_tensor(image)
x = TF.resize(x,[128,128])
# x = x*2-1.
x.unsqueeze_(0)
print(x.shape)
M_N = 0.111  # for the loss

def main():
    # Model
    print('==> Building model..')
    net = VanillaVAE(in_channels=3, latent_dim=args.latent_dim)
    net = net.to(device)

    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        # Load checkpoint.
        print('==> Resuming from checkpoint, loaded..')

    else:
        print('==> No checkpoint, return.')
        return None
    print("===> start evaluating ...")
    generate_images(net, x, name="test_reconstruct")
    # sample_images(net, name="test_randsample")


def train(net, trainloader, optimizer, criterion):
    net.train()
    train_loss = 0
    recons_loss = 0
    kld_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        result = net(inputs)
        loss_dict = criterion(result)  # loss, Reconstruction_Loss, KLD
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recons_loss += (loss_dict['Reconstruction_Loss']).item()
        kld_loss += (loss_dict['KLD']).item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Rec_Loss: %.3f | KLD_Loss: %.3f'
                     % (train_loss / (batch_idx + 1), recons_loss / (batch_idx + 1), kld_loss / (batch_idx + 1)))

    return {
        "train_loss": train_loss / (batch_idx + 1),
        "recons_loss": recons_loss / (batch_idx + 1),
        "kld_loss": kld_loss / (batch_idx + 1)
    }


def generate_images(net, x, name="val"):
    with torch.no_grad():
        img = x.to(device)
        vutils.save_image(img.float(), "inputs.png", nrow=1)

        out = net(img)
        recons = out["reconstruct"]
        mu = out["mu"]
        log_var = out["log_var"]
        print(recons.max(), recons.min(), recons.shape)
        # img = (img+1.0)/2.0
        # recons = (recons+1.0)/2.0
        vutils.save_image(recons.float(), "recons.png", nrow=1)

        # save_binary_img(recons.data,
        #                 os.path.join(args.checkpoint, f"{name}.png"),
        #                 nrow=args.val_num)


def sample_images(net, name="val"):
    with torch.no_grad():
        z = torch.randn(args.val_num, args.latent_dim)
        z = z.to(device)
        sampled = net.sample(z)
        save_binary_img(sampled.data,
                        os.path.join(args.checkpoint, f"{name}.png"),
                        nrow=args.val_num)


if __name__ == '__main__':
    main()
