---
typora-copy-images-to: ../fintune
---

# report about fintune

# settings

1. Model: mobile_v2(pertained in torchvision)
2. Dataset: food101,bird200(CUB), car196, cifar10, cifar100
3. pixel: 400*400



# usage

```
python3 mobilev2_imagenet.py path --device cuda --batchsize 64 --pretrain 1 -l 0.001
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

5. sum up

Tune classifier first (about 10epochs)

Use different learning rate on each layer,deep layer extract abstract informations.

Finetune all layers, lr<=1e-4


