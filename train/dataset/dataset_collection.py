import torch
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
class CUBDataset(Dataset):
    base_folder = 'CUB_200_2011/images'
    def __init__(self,root,train=True,transform = None, loader = default_loader, download=False,data=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.data =data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
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
        elif(self.type == 'CUB200'):
            images = pd.read_csv(os.path.join(self.path, 'CUB_200_2011', 'images.txt'), sep=' ',
                        names=['img_id', 'filepath'])
            image_class_labels = pd.read_csv(os.path.join(self.path, 'CUB_200_2011', 'image_class_labels.txt'),
                                    sep=' ', names=['img_id', 'target'])
            train_test_split = pd.read_csv(os.path.join(self.path,'CUB_200_2011', 'train_test_split.txt'),
                                            sep=' ', names=['img_id', 'is_training_img'])
            data = images.merge(image_class_labels, on='img_id')
            data = data.merge(train_test_split, on='img_id')
            train = data[data['is_training_img'].isin([1])]
            test = data[data['is_training_img'].isin([0])]
            train_dataset = CUBDataset(self.path,data = train,transform= self.compose['train'],)
            val_dataset = CUBDataset(self.path,data = test,transform= self.compose['val'],)
            return train_dataset,val_dataset
        elif(self.type == 'CIFAR10'):
            train_dataset = datasets.CIFAR10(root=self.path,train= True,transform= self.compose['train'],download=True)
            val_dataset = datasets.CIFAR10(root=self.path,train= False,transform= self.compose['val'])
            return train_dataset,val_dataset
        elif(self.type == 'Place365'):
            train_dataset = datasets.Places365(root=self.path, download = False,small = True,transform= self.compose['train'])
            val_dataset = datasets.Places365(root=self.path, download = False,small = True,transform= self.compose['val'])
            return train_dataset,val_dataset





