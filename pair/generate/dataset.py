import os
import pickle
import glob
import numpy as np
from torch.utils.data import Dataset


def load_data(root="../data/generate/generate/row_data"):
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
                all_loss.append(row["pixelwise_loss"].astype('float32'))
                all_segnum.append(row["best_num_segments"]-3) # minus 3 because it starts from 3
                all_color.append(color_map[row["best_color"]])


    all_loss = np.concatenate(all_loss, axis=0)
    all_segnum = np.concatenate([all_segnum], axis=0)
    all_color = np.concatenate([all_color], axis=0)
    return all_loss, all_segnum, all_color


class LossDataset(Dataset):
    def __init__(self, root="../data/generate/generate/row_data"):
        self.data, self.segnum, self.color = load_data(root)

    def __getitem__(self, item):
        data = self.data[item]
        print(data.shape)
        segnum = self.segnum[item]
        color = self.color[item]
        return data, segnum, color

    def __len__(self):
        return self.data.shape[0]

if __name__ == "__main__":
    load_data()
    dataset = LossDataset()
    print(dataset.__len__())
    print(dataset.__getitem__(28))

