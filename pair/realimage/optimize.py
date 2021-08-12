"""
CUDA_VISIBLE_DEVICES=0 python optimize.py --learning_rate 1.0 --optimizer sgd --msg sgd_lr1.0
SGD     lr 1.0      loss 0.078
SGD     lr 5.0      loss 0.061
SGD     lr 10.0     loss 0.029
SGD     lr 15.0     loss 0.032

Adam    lr 0.05     loss 0.008
Adam    lr 0.1      loss 0.007
Adam    lr 0.5      loss 0.014
Adam    lr 1.0      loss 0.021
Adam    lr 5.0      loss 0.135

"""
import argparse
import os
from PIL import Image
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
    parser.add_argument('--model', default='RealOptimize', help='model name [default: pointnet_cls]')
    parser.add_argument('--img_path', default='single.png')
    parser.add_argument('--optimizer', default='sgd')
    # training
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--frequency', default=100, type=int, help='workers')
    parser.add_argument('--loss', default='l2')
    # models
    # imsize = 28, paths = 4, segments = 5, samples = 2, zdim = 1024, stroke_width = None
    parser.add_argument('--imsize', default=224, type=int)
    parser.add_argument('--paths', default=128, type=int)
    parser.add_argument('--segments', default=3, type=int)
    parser.add_argument('--samples', default=2, type=int)
    parser.add_argument('--max_width', default=2, type=int)

    return parser.parse_args()


args = parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Verbose operations: make folder, init logger,  fix seed, set device
time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
message = time_str if args.msg is None else "-" + args.msg
args.checkpoint = 'checkpoints/' + args.model + message
args.visualize = 'checkpoints/' + args.model + message + '/visualize'
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)
if not os.path.isdir(args.visualize):
    mkdir_p(args.visualize)
screen_logger = logging.getLogger("Model")
screen_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler(os.path.join(args.checkpoint, "screen_out.txt"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
screen_logger.addHandler(file_handler)


def printf(str):
    screen_logger.info(str)
    print(str)


def main():
    if args.seed is not None:
        set_seed(args.seed)
        printf(f'==> fixing the random seed to: {args.seed}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    printf(f'==> using device: {device}')
    printf(f"==> args: {args}")

    # building models
    printf(f'==> Building model: {args.model}')
    net = models.__dict__[args.model](
        imsize=args.imsize, paths=args.paths, segments=args.segments, samples=args.samples, max_width=2)
    if args.loss == 'l1':
        criterion = nn.L1Loss().to(device)
        printf(f"Using criterion L1 loss.")
    else:
        criterion = nn.MSELoss().to(device)
        printf(f"Using criterion MSE loss.")

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    best_test_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="main")
        logger.set_names(["Epoch-Num", 'Learning-Rate', 'Train-Loss'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_test_loss = checkpoint['best_test_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="main", resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')
    transform = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    printf(f"==> Loading image: {args.img_path}")
    image = Image.open(args.img_path)
    if image.size()[0] == 4:
        # image = image[:3, :, :]  # remove alpha channel
        image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    printf(f"==> Saving the input image.")
    first_img = (image[0]).permute(1, 2, 0).cpu().numpy()
    plt.imsave(os.path.join(args.visualize, "input.png"), first_img)

    if args.optimizer == "sgd":
        printf("==> Using SGD optimizer")
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate,
                                    momentum=0.9, weight_decay=args.weight_decay)
    else:
        printf("==> Using Adam optimizer")
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch=start_epoch - 1)
    # init save images
    # visualize(net, train_loader, device, args.checkpoint +'/epoch-0', nrow=8)
    visualize(net, "init")
    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, image, optimizer, criterion)  # {"loss"}
        if (epoch + 1) % args.frequency == 0:
            visualize(net, epoch)
        scheduler.step()

        if train_out["loss"] < best_test_loss:
            best_test_loss = train_out["loss"]
            is_best = True
        else:
            is_best = False

        save_model(net, epoch, path=args.checkpoint, is_best=is_best, best_test_loss=best_test_loss,
                   test_loss=train_out["loss"], optimizer=optimizer.state_dict())
        logger.append([epoch, optimizer.param_groups[0]['lr'], train_out["loss"]])

        printf(f"Train loss:{train_out['loss']} [best loss:{best_test_loss}]")

    logger.close()


def train(net, image, optimizer, criterion):
    net.train()
    optimizer.zero_grad()
    out = net(image)
    loss = criterion(image, out)
    loss.backward()
    optimizer.step()
    return {
        "loss": float("%.3f" % (loss.item()))
    }


def visualize(net, epoch):
    net.eval()
    svgpath = os.path.join(args.visualize, f"epoch_{epoch}_svg.svg")
    renderpath = os.path.join(args.visualize, f"epoch_{epoch}_render.png")
    with torch.no_grad():
        net.module.visualize(svgpath=svgpath, renderpath=renderpath)
    printf(f"Finish visualization of epoch {epoch}.")


if __name__ == '__main__':
    main()
