import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_dataset_item(file_path):
    with np.load(file_path) as data:
        data_array = data['data']
        label_array = data['label']
    return data_array, label_array


class Prepped_Painting_Dataset(Dataset):
    def __init__(self, data_dir: str):

        self.data_dir = data_dir
        self.filenames = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.filenames[idx])

        data_array, label_array = load_dataset_item(data_path)

        return np.squeeze(data_array), label_array