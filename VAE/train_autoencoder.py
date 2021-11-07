from __future__ import print_function

import torch
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import argparse
from utils import mkdir_p, Logger, progress_bar, save_model, save_binary_img
from torchvision.datasets import ImageFolder
from autoencoder import AutoEncoder


parser = argparse.ArgumentParser(description='VAE training for NSF project')

# General MODEL parameters
parser.add_argument('--latent_dim', default=512, type=int)
# Parameters for  dataset
parser.add_argument('--traindir', default='../pair/data/emoji_rgb/train/', type=str, metavar='PATH',
                    help='path to training set')
parser.add_argument('--testdir', default='../pair/data/emoji_rgb/validate', type=str, metavar='PATH',
                    help='path to testing set')

# Parameters for  training
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--es', default=500, type=int, help='epoch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=64, type=int, help='batch size, better to have a square number')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--scheduler_gamma', default=0.985, type=float)


args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/%s_%s_bs%s' % (
    "AutoEncoder", args.latent_dim,args.bs)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
transform_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), SetRange])
transform_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), SetRange])
print('==> Preparing data..')
train_dataset = torchvision.datasets.ImageFolder(root=args.traindir, transform=transform_train)
test_dataset = torchvision.datasets.ImageFolder(root=args.testdir, transform=transform_test)

# data loader
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)

def main():
    start_epoch = 0
    best_loss = 9999999.99

    # Model
    print('==> Building model..')
    net = AutoEncoder(in_channels=3, latent_dim=args.latent_dim)
    net = net.to(device)

    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.es, eta_min=args.lr / 100)
    criterion = torch.nn.MSELoss

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
            print('==> Resuming from checkpoint, loaded..')
        else:
            print("==> No checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    if not args.evaluate:
        # training
        print("==> start training..")
        for epoch in range(start_epoch, args.es):
            print('\nEpoch: %d | Learning rate: %f ' % (epoch + 1, scheduler.get_last_lr()[-1]))
            train_out = train(net, trainloader, optimizer,criterion)  # {train_loss, recons_loss, kld_loss}
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint.pth'))
            if train_out["train_loss"] < best_loss:
                save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint_best.pth'),
                           loss=train_out["train_loss"])
                best_loss = train_out["train_loss"]
            logger.append([epoch + 1, scheduler.get_last_lr()[-1],
                           train_out["train_loss"]])
            scheduler.step()
        logger.close()
        print(f"\n==> Finish training..\n")

    # print("===> start evaluating ...")
    # generate_images(net, testloader, name="test_reconstruct")
    # sample_images(net, name="test_randsample")


def train(net, trainloader, optimizer, criterion):
    net.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        result,_ = net(inputs)
        loss = criterion(result, inputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f '  % (train_loss / (batch_idx + 1)))

    return {
        "train_loss": train_loss / (batch_idx + 1)
    }


def generate_images(net, valloader, name="val"):
    dataloader_iterator = iter(valloader)
    with torch.no_grad():
        img, spe = next(dataloader_iterator)
        img = img.to(device)
        recons = net.generate(img)
        result = torch.cat([img, recons], dim=0)
        save_binary_img(result.data,
                        os.path.join(args.checkpoint, f"{name}.png"),
                        nrow=args.val_num)


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
