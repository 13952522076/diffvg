'''Train LossData with PyTorch.
e.g.
    python3 train.py
'''
from __future__ import print_function

import numpy as np
import torch
import shutil
import logging
import datetime
import subprocess
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from resnet_se import se_resnet18
import torchvision
from dataset import LossDataset
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--root', default="../data/generate/generate/row_data", type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--workers', default=8, type=int, help='batch size')
    parser.add_argument('--epoch', default=300, type=int, help='epoch size')
    parser.add_argument('--normalize', type=bool,  default=True, help='normalize input data')
    parser.add_argument('--alpha', default=2.0, type=float, help='learning rate')
    return parser.parse_args()


def get_git_commit_id():
    try:
        cmd_out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        return cmd_out
    except:
        # indicating no git found.
        return "0000000"


def focal_loss(outputs, targets, alpha=1, gamma=5):
    ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') # important to add reduction='none' to keep per-batch-item loss
    # print(f"ce_loss shape is: {ce_loss.shape}")
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean() # mean over the batch
    return focal_loss

def regression_loss(outputs, targets, weight=np.array([0.26, 0.87, 0.95, 0.96, 0.96, 0.97])):
    # print(outputs)
    # loss = (abs(outputs - targets))
    # pt = torch.exp(loss*weight)
    # # print(loss.shape)
    # return (pt*loss).mean()

    weights = torch.Tensor(weight[targets.cpu().numpy()]).to(outputs.device)
    loss = (abs(outputs - targets))*(weights**3)
    return loss.mean()

def save_model(net, epoch, path, acc, is_best, **kwargs):
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'acc': acc
    }
    for key, value in kwargs.items():
        state[key] = value
    filepath = os.path.join(path, "last_checkpoint.pth")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best_checkpoint.pth'))

def main():
    args = parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    time_str = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    args.checkpoint = 'checkpoints/' + time_str
    if not os.path.isdir(args.checkpoint):
        try:
            os.makedirs(args.checkpoint)
        except OSError as exc:  # Python >2.5
            pass
    # logger file
    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    printf(f"args: {args}")
    printf(f"\n=== Current git ID is: {get_git_commit_id()} ===\n")

    # Model
    printf('==> Building model, criterion, optimizer..')
    net = se_resnet18().to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # criterion = nn.CrossEntropyLoss()
    # criterion = focal_loss
    criterion = regression_loss
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.lr / 1000)
    best_test_acc = 0.  # best test accuracy
    start_epoch = 0

    # Data
    printf('==> Preparing data..')
    # Data
    transform_train = transforms.Compose([ transforms.ToPILImage(), transforms.Resize(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([ transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor() ])
    trainset = LossDataset(root=os.path.join(args.root, "train"),  normalize=args.normalize, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    testset = LossDataset(root=os.path.join(args.root, "test"),  normalize=args.normalize, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.workers)

    for epoch in range(start_epoch, args.epoch):
        printf('\n\nEpoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        # {"loss", "loss_segnum", "loss_color", "acc_segnum", "acc_color", "time"}
        train_out = train(net, train_loader, optimizer, criterion, device, args)
        test_out = validate(net, test_loader, criterion, device, args)
        scheduler.step()
        if 0.5*(test_out["acc_color"] + test_out["acc_segnum"]) > best_test_acc:
            best_test_acc = 0.5*(test_out["acc_color"] + test_out["acc_segnum"])
            is_best = True
        else:
            is_best = False
            {"loss", "loss_segnum", "acc_segnum", "acc_color", "time"}
        printf(f'Training loss: {train_out["loss"]}, loss_segnum: {train_out["loss_segnum"]}, '
               f'loss_color: {train_out["loss_color"]},'
               f'acc_segnum: {train_out["acc_segnum"]}, acc_color: {train_out["acc_color"]}')
        printf(f'Testing  loss: {test_out["loss"]}, loss_segnum: {test_out["loss_segnum"]}, '
               f'loss_color: {test_out["loss_color"]}, acc_segnum: {test_out["acc_segnum"]},'
               f' acc_color: {test_out["acc_color"]}, best_meanAcc_test: {best_test_acc}')
        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["acc_color"] + test_out["acc_segnum"],
            is_best=is_best, best_test_acc=best_test_acc,  # best test accuracy
            acc_color = test_out["acc_color"],
            acc_segnum = test_out["acc_segnum"],

            optimizer=optimizer.state_dict()
        )


def train(net, trainloader, optimizer, criterion, device, args):
    net.train()
    train_loss = 0
    train_loss_segnum = 0
    train_loss_color = 0
    correct_segnum = 0
    correct_color = 0
    total = 0
    time_cost = datetime.datetime.now()
    t_range = tqdm(trainloader)
    for batch_idx, (data, label_segnum, label_color) in enumerate(t_range):
        data, label_segnum, label_color  = data.to(device), label_segnum.to(device), label_color.to(device)
        logits_segnum, logits_color = net(data)

        loss_segnum = criterion(logits_segnum, label_segnum, np.array([0.26, 0.87, 0.95, 0.96, 0.96, 0.97]))
        loss_color = args.alpha * criterion(logits_color, label_color, np.array([0.20, 0.89, 0.89]))
        loss = loss_segnum + loss_color
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        train_loss_segnum += loss_segnum.item()
        train_loss_color +=loss_color.item()
        preds_segnum = logits_segnum.max(dim=1)[1]
        preds_color = logits_color.max(dim=1)[1]

        total += label_segnum.size(0)
        correct_segnum += preds_segnum.eq(label_segnum).sum().item()
        correct_color += preds_color.eq(label_color).sum().item()

        t_range.set_postfix({'Train loss':(train_loss / (batch_idx + 1)),
                             "loss_segnum": (train_loss_segnum / (batch_idx + 1)),
                             "loss_color": (train_loss_color / (batch_idx + 1)),
                             "acc_segnum": correct_segnum/total,
                             "acc_color": correct_color/total,
                             })
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    print(f"Train max preds_segnum is: {(preds_segnum).max()}, max preds_color is: {preds_color.max()}")
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "loss_segnum": float("%.3f" % (train_loss_segnum / (batch_idx + 1))),
        "loss_color": float("%.3f" % (train_loss_color / (batch_idx + 1))),
        "acc_segnum": float("%.3f" % (100. * correct_segnum/total)),
        "acc_color": float("%.3f" % (100. * correct_color/total)),
        "time": time_cost
    }


def validate(net, valloader, criterion, device, args):
    net.eval()
    val_loss = 0
    val_loss_segnum = 0
    val_loss_color = 0
    correct_segnum = 0
    correct_color = 0
    total = 0
    time_cost = datetime.datetime.now()
    t_range = tqdm(valloader)
    with torch.no_grad():
        for batch_idx, (data, label_segnum, label_color) in enumerate(t_range):
            data, label_segnum, label_color  = data.to(device), label_segnum.to(device), label_color.to(device)
            logits_segnum, logits_color = net(data)
            loss_segnum = criterion(logits_segnum, label_segnum, np.array([0.26, 0.87, 0.95, 0.96, 0.96, 0.97]))
            loss_color = args.alpha * criterion(logits_color, label_color, np.array([0.20, 0.89, 0.89]))
            loss = loss_segnum + loss_color
            val_loss += loss.item()
            val_loss_segnum += loss_segnum.item()
            val_loss_color +=loss_color.item()
            preds_segnum = logits_segnum.max(dim=1)[1]
            preds_color = logits_color.max(dim=1)[1]

            total += label_segnum.size(0)
            correct_segnum += preds_segnum.eq(label_segnum).sum().item()
            correct_color += preds_color.eq(label_color).sum().item()

            t_range.set_postfix({'Val loss':(val_loss / (batch_idx + 1)),
                                 "loss_segnum": (val_loss_segnum / (batch_idx + 1)),
                                 "loss_color": (val_loss_color / (batch_idx + 1)),
                                 "acc_segnum": correct_segnum/total,
                                 "acc_color": correct_color/total,
                                 })
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    print(f"Valid max preds_segnum is: {(preds_segnum).max()}, max preds_color is: {preds_color.max()}")
    return {
        "loss": float("%.3f" % (val_loss / (batch_idx + 1))),
        "loss_segnum": float("%.3f" % (val_loss_segnum / (batch_idx + 1))),
        "loss_color": float("%.3f" % (val_loss_color / (batch_idx + 1))),
        "acc_segnum": float("%.3f" % (100. * correct_segnum/total)),
        "acc_color": float("%.3f" % (100. * correct_color/total)),
        "time": time_cost
    }








if __name__ == '__main__':
    main()
