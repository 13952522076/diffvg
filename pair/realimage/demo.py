
from torch.utils.data import DataLoader
# from models.RealAE import RealAE
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F



# transform = transforms.Compose([
#         transforms.Resize(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()
#     ])
#
# dataset = ImageFolder(root="./data/emoji/", transform=transform)
# train_loader = DataLoader(dataset, num_workers=4, batch_size=4, shuffle=True, pin_memory=False)
# for batch_idx, (data, label) in enumerate(train_loader):
#     first_img = (data[0]).permute(1,2,0).numpy()
#     print(first_img.shape)
#     plt.imsave("test.png",first_img)
#     plt.show()





a = torch.rand([2,3,28,28])
b = torch.rand([2,3,28,28])
loss1 = F.mse_loss(a, b)
loss2 = (a-b).pow(2).mean()
print(loss1)
print(loss2)
