"""
CUDA_VISIBLE_DEVICES=0 python visualize.py --model ResNetAE --msg demo1 --image ../data/emoji_rgb/train/0/240px-Emoji_u1f60d.svg.png
"""
import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
import models as models
from PIL import Image
from helper import mkdir_p


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--model', default='RealAE', help='model name [default: pointnet_cls]')
    parser.add_argument('--image', type=str, help='the test image')
    parser.add_argument('--which', type=str, default="best", choices=["best", "last"])

    # training
    parser.add_argument('--loss', default='l2')

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
args.visualize = 'checkpoints/' + args.model + message + '/test'
if not os.path.isdir(args.visualize):
    mkdir_p(args.visualize)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==> using device: {device}')
    print(f"==> args: {args}")

    # building models
    print(f'==> Building model: {args.model}')
    net = models.__dict__[args.model](
        imsize=args.imsize, paths=args.paths, segments=args.segments, samples=args.samples,
        zdim=args.zdim, pretained_encoder=args.pretained_encoder)

    if args.loss == 'l1':
        criterion = nn.L1Loss().to(device)
        print(f"==> Using criterion L1 loss.")
    else:
        criterion = nn.MSELoss().to(device)
        print(f"==> Using criterion MSE loss.")

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    which_checkpoint = args.which + "_checkpoint.pth"
    print(f"==> loading {which_checkpoint} from {args.checkpoint}")
    checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])

    print('==> Preparing data..')
    test_transform = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        transforms.ToTensor()
    ])
    data = Image.open(args.image)
    data = test_transform(data)
    data.unsqueeze_(0)
    data = data.to(device)

    net.eval()
    basename = os.path.basename(args.image)
    filename = os.path.splitext(basename)[0]
    print(f"basename is {basename}, filename is{filename}")

    with torch.no_grad():
        out = net(data)
        loss = criterion(data, out)
        loss = str("%.6f" % (loss))

        inputpath = os.path.join(args.visualize, f"{filename}_input.png")
        svgpath = os.path.join(args.visualize, f"{filename}_loss{loss}_svg.svg")
        renderpath = os.path.join(args.visualize, f"{filename}_loss{loss}_render.png")

        net.module.visualize(data, inputpath=inputpath, svgpath=svgpath, renderpath=renderpath)
    print(f"Finish visualization.")

if __name__ == '__main__':
    main()
