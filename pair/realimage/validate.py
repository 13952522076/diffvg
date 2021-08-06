import argparse
import os
import logging
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torchvision.datasets.mnist import FashionMNIST, MNIST
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
from tqdm import tqdm
import models as models
import matplotlib.pyplot as plt
from helper import mkdir_p, save_model, save_args, set_seed, Logger


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--model', default='RealAE', help='model name [default: pointnet_cls]')
    # training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    parser.add_argument('--frequency', default=5, type=int, help='workers')
    parser.add_argument('--loss', default='l1')
    # models
    # imsize = 28, paths = 4, segments = 5, samples = 2, zdim = 1024, stroke_width = None
    parser.add_argument('--imsize', default=224, type=int)
    parser.add_argument('--paths', default=128, type=int)
    parser.add_argument('--segments', default=3, type=int)
    parser.add_argument('--samples', default=2, type=int)
    parser.add_argument('--zdim', default=2048, type=int)
    parser.add_argument('--max_width', default=2, type=int)
    parser.add_argument('--pretained_encoder', dest='pretained_encoder', action='store_true')



    return parser.parse_args()


args = parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Verbose operations: make folder, init logger,  fix seed, set device
time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
message = time_str if args.msg is None else "-" + args.msg
args.checkpoint = 'checkpoints/' + args.model + message
args.visualize = 'checkpoints/' + args.model + message +'/visualize_validate'
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)
if not os.path.isdir(args.visualize):
    mkdir_p(args.visualize)


def main():

    if args.seed is not None:
        set_seed(args.seed)
        print(f'==> fixing the random seed to: {args.seed}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==> using device: {device}')
    print(f"==> args: {args}")

    # building models
    print(f'==> Building model: {args.model}')
    net = models.__dict__[args.model](
        imsize=args.imsize, paths=args.paths, segments=args.segments, samples=args.samples,
        zdim=args.zdim, max_width=2, pretained_encoder=args.pretained_encoder)
    if args.loss == 'l1':
        criterion = nn.L1Loss().to(device)
        print(f"Using criterion L1 loss.")
    else:
        criterion = nn.MSELoss().to(device)
        print(f"Using criterion MSE loss.")

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    print(f"Resuming last checkpoint from {args.checkpoint}")
    checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    print(f"load epoch-{start_epoch} checkpoint.")


    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.Resize((args.imsize,args.imsize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root="./data/full_emoji/", transform=transform)
    test_loader = DataLoader(dataset, num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=True, pin_memory=False)

    test_out = validate(net, test_loader, criterion, device)  # {"loss"}
    print(f"Test loss is: {test_out['loss']}")
    visualize(net, test_loader, device)


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(tqdm(testloader)):
            data, label = data.to(device), label.to(device)
            out = net(data)
            loss = criterion(out, data)
            test_loss += loss.item()
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1)))
    }

def visualize(net, testloader, device):
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(tqdm(testloader)):
            inputpath = os.path.join(args.visualize, f"validate_{batch_idx}_input.png")
            svgpath = os.path.join(args.visualize, f"validate_{batch_idx}_svg.svg")
            renderpath = os.path.join(args.visualize, f"validate_{batch_idx}_render.png")
            data, label = data.to(device), label.to(device)
            net.module.visualize(data, inputpath=inputpath, svgpath=svgpath, renderpath=renderpath)
    print(f"Finish visualization.")



if __name__ == '__main__':
    main()
