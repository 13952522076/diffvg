from __future__ import print_function
import argparse
import torch
# import cv2
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--evaluate', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--img_1', type=int, default=1)
parser.add_argument('--img_2', type=int, default=4)
parser.add_argument('--img_3', type=int, default=30)
parser.add_argument('--img_4', type=int, default=60)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 64)
        self.fc22 = nn.Linear(400, 64)
        self.fc3 = nn.Linear(64, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    if not args.evaluate:
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            test(epoch)
            torch.save(model.state_dict(),"last.pth")
            scheduler.step()
            with torch.no_grad():
                sample = torch.randn(64, 64).to(device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28),
                           'results/sample_' + str(epoch) + '.png')
    else:
        dataset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
        model.load_state_dict(torch.load("last.pth",map_location=torch.device('cpu')))
        data1, label1 = dataset.__getitem__(args.img_1)
        data2, label2 = dataset.__getitem__(args.img_2)
        data3, label3 = dataset.__getitem__(args.img_3)
        data4, label4 = dataset.__getitem__(args.img_4)
        model.eval()
        with torch.no_grad():
            # [1 2]
            # [3 4]
            _, mu1, logvar1 = model(data1.view(1, 1, 28, 28))
            _, mu2, logvar2 = model(data2.view(1, 1, 28, 28))
            _, mu3, logvar3 = model(data3.view(1, 1, 28, 28))
            _, mu4, logvar4 = model(data4.view(1, 1, 28, 28))
            for i in [0, 0.1,  0.3,  0.5,  0.7, 0.9, 1.0]:
                for j in [0, 0.1,  0.3,  0.5,  0.7, 0.9, 1.0]:
                    # sample interplolated mean std
                    mu_row_1 = (1-i)*mu1+ i*mu2
                    logvar_row_1 = (1-i)*logvar1+ i*logvar2
                    mu_row_2 = (1-i)*mu3+ i*mu4
                    logvar_row_2 = (1-i)*logvar3+ i*logvar4

                    mu_col = (1-j)*mu_row_1+ j*mu_row_2
                    logvar_col=(1-j)*logvar_row_1+ j*logvar_row_2


                    hidden = model.reparameterize(mu_col,logvar_col)

                    sampled = 1 - model.decode(hidden).view(1, 1, 28, 28) # change the color
                    # remove some noise
                    # sampled[sampled>=0.5] = 1.0
                    # sampled[sampled<0.5] = 0.0
                    save_image(sampled, f'matrix/id{args.img_1}_id{args.img_2}_id{args.img_3}_id{args.img_4}_{i}_{j}.png')


        out = 1-data1.view(1, 1, 28, 28)
        # out[out>=0.5] = 1.0
        # out[out<0.5] = 0.0
        save_image(out, 'results/data1.png')
        out = 1-data2.view(1, 1, 28, 28)
        # out[out>=0.5] = 1.0
        # out[out<0.5] = 0.0
        save_image(out, 'results/data2.png')
        # print(data2)
        print(label1, label2, label3, label4)
        #### hr https://bigjpg.com/
