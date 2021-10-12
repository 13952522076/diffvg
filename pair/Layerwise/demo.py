
import torch

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings
warnings.filterwarnings("ignore")



num_iter = 1000
background_vars = [torch.rand([4],requires_grad=True)]
back_optim = torch.optim.Adam(background_vars, lr=0.01)
target = torch.ones([4],requires_grad=False)
scheduler = CosineAnnealingLR(back_optim, num_iter, eta_min=0.0001)
for t in tqdm(range(num_iter)):
    back_optim.zero_grad()
    loss = (abs(target-background_vars[0])).sum()
    loss.backward()
    back_optim.step()
    scheduler.step()
    print(f"{background_vars}  {loss}")
print((background_vars[0]).shape)

