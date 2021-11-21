# report about fintune

# settings

1. Model: mobile_v2(pertained in torchvision)
2. Dataset: food101,bird200(CUB), car196, cifar10, cifar100
3. pixel: 400*400



# usage

```
usage: train.py [-h] [-a ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
                [--lr LR] [--momentum M] [--wd W] [-p N] [--resume PATH] [-e]
                [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
                [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
                [-t DATASET_TYPE] [--gamma GAMMA]
                DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        efficientnet_b0 | efficientnet_b1 | efficientnet_b2 |
                        efficientnet_b3 | efficientnet_b4 | efficientnet_b5 |
                        efficientnet_b6 | efficientnet_b7 | googlenet |
                        inception_v3 | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 |
                        mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large |
                        mobilenet_v3_small | regnet_x_16gf | regnet_x_1_6gf |
                        regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf |
                        regnet_x_800mf | regnet_x_8gf | regnet_y_16gf |
                        regnet_y_1_6gf | regnet_y_32gf | regnet_y_3_2gf |
                        regnet_y_400mf | regnet_y_800mf | regnet_y_8gf |
                        resnet101 | resnet152 | resnet18 | resnet34 | resnet50
                        | resnext101_32x8d | resnext50_32x4d |
                        shufflenet_v2_x0_5 | shufflenet_v2_x1_0 |
                        shufflenet_v2_x1_5 | shufflenet_v2_x2_0 |
                        squeezenet1_0 | squeezenet1_1 | vgg11 | vgg11_bn |
                        vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn
                        | wide_resnet101_2 | wide_resnet50_2 (default:
                        resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  -p N, --print-freq N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  -t DATASET_TYPE, --dataset-type DATASET_TYPE
                        choose a dataset to train
  --gamma GAMMA         decay rate at scheduler
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




