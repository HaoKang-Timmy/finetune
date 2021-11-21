import torch
import os
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
class DatasetCollection():
    def __init__(self,type,path,compose_train,compose_val):
        self.type = type
        self.path = path
        self.compose = {}
        self.compose['train'] = compose_train
        self.compose['val'] = compose_val
    def init(self):
        if(self.type == 'Imagenet'):
            traindr = os.path.join(self.path, 'train')
            valdr = os.path.join(self.path, 'val')
            train_dataset = datasets.ImageFolder(
                traindr,
                self.compose['train']
            )
            val_dataset = datasets.ImageFolder(
                valdr,
                self.compose['val']
            )
            return train_dataset,val_dataset



