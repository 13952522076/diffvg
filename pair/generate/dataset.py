import os
import pickle
import glob
import numpy as np
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple


def load_data(root="../data/generate/generate/row_data/train", normalize=True):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_loss = []
    all_segnum = []
    all_color = []
    color_map= {"RadialGradient":0, "LinearGradient":1, "Normal":2}
    for file_name in glob.glob(os.path.join(BASE_DIR, root, '*.pkl')):
        with open(file_name, "rb") as f:
            row_list = pickle.load(f)
            for row in row_list:
                data = row["pixelwise_loss"].astype('float32')
                if normalize:
                    data = (data-data.mean())/(data.std() + 1e-8)  # mean/std normalize
                    data -= data.min()
                    data /= (data.max() + 1e-8) # scale to [0, 1]

                all_loss.append(data)
                all_segnum.append(row["best_num_segments"]-3) # minus 3 because it starts from 3
                all_color.append(color_map[row["best_color"]])


    all_loss = np.concatenate(all_loss, axis=0)
    all_segnum = np.concatenate([all_segnum], axis=0)
    all_color = np.concatenate([all_color], axis=0)
    return all_loss, all_segnum, all_color


class LossDataset(Dataset):
    def __init__(self, root="../data/generate/generate/row_data/train", normalize=True, transform: Optional[Callable] = None):
        self.data, self.segnum, self.color = load_data(root, normalize)
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[item]
        data = data.transpose(1, 2, 0)
        segnum = self.segnum[item]
        color = self.color[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, segnum, color

    def __len__(self):
        return self.data.shape[0]

if __name__ == "__main__":
    import torchvision.transforms as transforms
    load_data()
    transform_train = transforms.Compose([ transforms.ToPILImage(),
        transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    dataset = LossDataset(transform=transform_train)

    Seg3_num = 0
    Seg4_num = 0
    Seg5_num = 0
    Seg6_num = 0
    Seg7_num = 0
    Seg8_num = 0
    Norm_num = 0
    Linear_num = 0
    Radial_num = 0
    for i in range(0, len(dataset.segnum)):
        if dataset.segnum[i] ==0:
            Seg3_num +=1
        if dataset.segnum[i] ==1:
            Seg4_num +=1
        if dataset.segnum[i] ==2:
            Seg5_num +=1
        if dataset.segnum[i] ==3:
            Seg6_num +=1
        if dataset.segnum[i] ==4:
            Seg7_num +=1
        if dataset.segnum[i] ==5:
            Seg8_num +=1
        if dataset.color[i] ==0:
            Radial_num +=1
        if dataset.color[i] ==1:
            Linear_num +=1
        if dataset.color[i] ==2:
            Norm_num +=1

    print(f"Seg3: {Seg3_num} Seg4: {Seg4_num} Seg5: {Seg5_num} Seg6: {Seg6_num} Seg7: {Seg7_num} Seg8: {Seg8_num}")
    print(f"Radial_num: {Radial_num} Linear_num: {Linear_num} Norm_num: {Norm_num}")


    # print(dataset.__len__())
    # print(dataset.__getitem__(28))
    for i in range (1,500):
        t = np.random.randint(0,300)
        data, segnum, color = dataset.__getitem__(t)
        # print(data.max(), data.min(), segnum, color)
    print(data.mean())

