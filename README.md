# report about fintune

# settings

1. Model: All models in torchvision(pertained in torchvision)
2. Dataset: food101,bird200(CUB), car196, cifar10, cifar100,Imagenet

# structure
```
.
├── README.md
├── pic
│   └── image-20211121141842034.png
└── train
    ├── dataset
    │   └── dataset_collection.py
    ├── train.py
    └── vision
        └── vision_class.py
```
vision_class.py for some helper functions and visualizition
dataset_collecton.py for generate datasets for food101,bird200(CUB), car196, cifar10, cifar100,Imagenet
# usage

```
usage: train.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
                [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e]
                [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
                [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
                [-t DATASET_TYPE] [--gamma GAMMA]
                DIR
```

# example
```
python train.py -a mobilenet_v2 --pretrained --dist-url 'tcp://127.0.0.1:1234' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0  --seed 1 -t Imagenet /dataset/imagenet
```

# results

| Dataset  | Val_acc%(top1)             | Test_acc%(top1) |
| -------- | -------------------------- | --------------- |
| Cifar10  | 96.224                     | 95.781          |
| Car196   | 87.295                     |                 |
| Food101  | 80.124(Image net bench=83) |                 |
| Cifar100 | 71.242(to be done)         | 70.411          |
| CUB200   | 72.141                     |                 |
| Imagenet | 71.095                     |                 |
# curve

![image-20211121141842034](./pic/image-20211121141842034.png)

# techniques

1. Learning rate decay

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.95)
......
scheduler.step()
```

2. weight decay

```
optimizer = torch.optim.Adam(net.module.parameters(),lr = lr_init,weight_decay=weight_decay)
```

prevent overfit in car and birds but not that useful in food

3. K-folder(should have test datas)

```
for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
  print('------------fold no---------{}----------------------'.format(fold))
  train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
  test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
 
  trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=batch_size, sampler=train_subsampler)
  testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=batch_size, sampler=test_subsampler)
```

4. different learning rate on each layer

```
optimizer = torch.optim.Adam([{'params':classifier_params},{'params':low_params,'lr':lr_init*0.6},{'params':deep_params,'lr':lr_init*0.4}],lr=lr_init)
```




