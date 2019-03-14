import torch
import numpy as np
from torch.utils import data
import glob

class Dataset(data.Dataset):
    def __init__(self, directory):
        self.files = glob.glob(directory)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        wav = torch.load(file_name)
        return wav

class RBMSampler(data.Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        return iter(np.random.choice(len(self.data_source), size=self.batch_size, replace=False).tolist())

    def __len__(self):
        return len(self.data_source)