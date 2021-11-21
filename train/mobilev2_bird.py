import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
import torchvision
class Mydataset(Dataset):

    base_folder = 'CUB_200_2011/images'
    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False,data=None):
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
root = '/home/haokang/fintune/data/bird'
batch =64
images = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'images.txt'), sep=' ',
                        names=['img_id', 'filepath'])
image_class_labels = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'image_class_labels.txt'),
                                    sep=' ', names=['img_id', 'target'])
train_test_split = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'train_test_split.txt'),
                                sep=' ', names=['img_id', 'is_training_img'])
data = images.merge(image_class_labels, on='img_id')

data = data.merge(train_test_split, on='img_id')

train = data[data['is_training_img'].isin([0])]
print(train)

test = data[data['is_training_img'].isin([1])]
print(test.shape[0])


transforms_train = transforms.Compose([transforms.RandomResizedCrop(400),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
trainds = Mydataset(root,data = train,transform=transforms_train)
testds = Mydataset(root,data = test,transform=transforms_train,train=False)
train_loader = DataLoader(trainds, batch_size = batch, shuffle=True)
valid_loader = DataLoader(testds, batch_size = batch)



num_epoch = 20
lr_init = 0.00001
weight_decay = 0.01
save_path = './model/mobilev2_bird.pth'



device = torch.device("cuda")
net = torchvision.models.mobilenet_v2(pretrained=True)
net.classifier[-1] = nn.Linear(1280,200)
net.load_state_dict(torch.load(save_path))
# for param in net.parameters():
#     param.requires_grad = True
classifier_map = list(map(id,net.classifier.parameters()))
low_map = list(map(id,net.features[-5:]))
classifier_params = filter(lambda p : id(p) in classifier_map,net.parameters())
low_params = filter(lambda p : id(p) in low_map,net.parameters())
deep_params = filter(lambda p : id(p) not in low_map+classifier_map,net.parameters())
for param in net.parameters():
    param.requires_grad = True
for param in net.classifier.parameters():
    param.requires_grad = True
for param in low_params:
    param.requires_grad = True
net = nn.DataParallel(net).to(device)
loss_function = nn.CrossEntropyLoss()

def train():
    best_acc = 0.70287
    #optimizer = torch.optim.Adam([{'params':classifier_params},{'params':low_params,'lr':lr_init*0.7},{'params':deep_params,'lr':lr_init*0.4}],lr=lr_init,weight_decay=weight_decay)
    optimizer = torch.optim.Adam(net.module.parameters(),lr = lr_init,weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.9)
    for i in range(num_epoch):
        #optimizer = torch.optim.Adam(net.module.parameters(),lr = lr_init,weight_decay=weight_decay)
        #optimizer = torch.optim.Adam([{'params':classifier_params},{'params':low_params,'lr':lr_init*0.6},{'params':deep_params,'lr':lr_init*0.4}],lr=lr_init)
        net.train()
        running_loss = 0.0
        
        for step, data in enumerate(train_loader, start=0):
            images,labels = data
            optimizer.zero_grad()
            images = images.to(device)
            logits = net(images)
            labels = torch.tensor(labels).to(device)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step+1)/(5700/batch)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print(
                "\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        #scheduler.step()
        net.eval()
        acc = 0.0
        test_num = 0.0
        with torch.no_grad():
            for val_data in valid_loader:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                outputs = net(val_images)  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels).sum().item()
                test_num += val_labels.shape[0]
                
            val_accurate = acc / test_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.module.state_dict(), save_path)
                print("save")
            print('[epoch %d] train_loss: %.5f  test_accuracy: %.5f' %
                (i + 1, running_loss / step, val_accurate))
train()