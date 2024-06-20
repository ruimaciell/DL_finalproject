#basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import xarray as xr
import dask.array as da

#preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.mixture import GaussianMixture

#parallelization
import concurrent.futures

#torch
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from PIL import Image


meta = pd.read_csv('/Users/mikelgallo/repos3/final_dl/wikiart-target_style-class_26-keepgenre_True-n_100_drop-raw.csv')
label_list = meta['style_path'].unique().tolist()
label_codes = {idx:val for idx, val in enumerate(label_list)}


class Painting_Dataset(Dataset):
    def __init__(self, data_dir:str, transform=None, target_transform=None):

        self.data_dir = data_dir
        self.filenames = sorted(os.listdir(data_dir))
        self.transform = transform
        self.target_transform = target_transform
        self.meta = meta
        self.label_codes = label_codes
    

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir,self.filenames[idx])
        img = Image.open(data_path).convert('RGB')
        image_name = os.path.basename(data_path)
        label = self.meta[self.meta['chan_image_name'] == self.filenames[idx]]['style_path'].item()
        label_enc = next(key for key, value in self.label_codes.items() if value == label)

        #img = self.processor.decode_img(img)
        #label = self.processor.get_label(label, self.class_names)

        if self.transform:
            img = self.transform(img)  # Transform the image to tensor
        
        if self.target_transform:
            label = self.target_transform(label_enc)

        return img, label_enc, image_name
