# report about fintune

# settings

1. Model: mobile_v2(pertained in torchvision)
2. Dataset: food101,bird200(CUB), car196, cifar10, cifar100
3. pixel: 400*400

# results

| Dataset  | Val_acc%(top1)             | Test_acc%(top1) |
| -------- | -------------------------- | --------------- |
| Cifar10  | 96.224                     | 95.781          |
| Car196   | 87.295                     |                 |
| Food101  | 80.124(Image net bench=83) |                 |
| Cifar100 | 71.242(to be done)         | 70.411          |
| CUB200   | 72.141                     |                 |

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

# sum up 

1. difference about `data parallel`and `distributed.dataparallel`

Later one is better, but can not be used in jupyter notebook.

2. Dataset could be divided into iterated and mapped

I use mapped dataset here.

3. structure understanding (to be done)

# log(cifar10)

```
0
train loss: 100%[**************************************************->]0.9549[epoch 1] train_loss: 0.859  test_accuracy: 0.874
1
train loss: 100%[**************************************************->]0.7479[epoch 2] train_loss: 0.656  test_accuracy: 0.909
2
train loss: 100%[**************************************************->]0.5570[epoch 3] train_loss: 0.585  test_accuracy: 0.916
3
train loss: 100%[**************************************************->]0.7522[epoch 4] train_loss: 0.545  test_accuracy: 0.910
4
train loss: 100%[**************************************************->]0.5303[epoch 5] train_loss: 0.510  test_accuracy: 0.918
5
train loss: 100%[**************************************************->]0.6774[epoch 6] train_loss: 0.489  test_accuracy: 0.923
6
train loss: 100%[**************************************************->]0.4580[epoch 7] train_loss: 0.454  test_accuracy: 0.923
7
train loss: 100%[**************************************************->]0.4754[epoch 8] train_loss: 0.446  test_accuracy: 0.930
8
train loss: 100%[**************************************************->]0.5665[epoch 9] train_loss: 0.434  test_accuracy: 0.932
9
train loss: 100%[**************************************************->]0.5517[epoch 10] train_loss: 0.421  test_accuracy: 0.937
10
train loss: 100%[**************************************************->]0.5131[epoch 11] train_loss: 0.410  test_accuracy: 0.941
11
train loss: 100%[**************************************************->]0.3803[epoch 12] train_loss: 0.393  test_accuracy: 0.944
12
train loss: 100%[**************************************************->]0.5413[epoch 13] train_loss: 0.382  test_accuracy: 0.943
13
train loss: 100%[**************************************************->]0.3089[epoch 14] train_loss: 0.378  test_accuracy: 0.943
14
train loss: 100%[**************************************************->]0.2076[epoch 15] train_loss: 0.369  test_accuracy: 0.945
15
train loss: 100%[**************************************************->]0.4347[epoch 16] train_loss: 0.361  test_accuracy: 0.948
16
train loss: 100%[**************************************************->]0.2480[epoch 17] train_loss: 0.356  test_accuracy: 0.947
17
train loss: 100%[**************************************************->]0.3702[epoch 18] train_loss: 0.349  test_accuracy: 0.948
18
train loss: 100%[**************************************************->]0.4546[epoch 19] train_loss: 0.341  test_accuracy: 0.951
19
train loss: 100%[**************************************************->]0.2457[epoch 20] train_loss: 0.332  test_accuracy: 0.948



0
train loss: 100%[**************************************************->]0.3469[epoch 1] train_loss: 0.337  test_accuracy: 0.940
1
train loss: 100%[**************************************************->]0.2476[epoch 2] train_loss: 0.332  test_accuracy: 0.936
2
train loss: 100%[**************************************************->]0.3756[epoch 3] train_loss: 0.329  test_accuracy: 0.937
3
train loss: 100%[**************************************************->]0.3371[epoch 4] train_loss: 0.326  test_accuracy: 0.943
4
train loss: 100%[**************************************************->]0.4395[epoch 5] train_loss: 0.316  test_accuracy: 0.949
5
train loss: 100%[**************************************************->]0.3204[epoch 6] train_loss: 0.317  test_accuracy: 0.940
6
train loss: 100%[**************************************************->]0.4194[epoch 7] train_loss: 0.312  test_accuracy: 0.938
7
train loss: 100%[**************************************************->]0.3217[epoch 8] train_loss: 0.311  test_accuracy: 0.942
8
train loss: 100%[**************************************************->]0.2354[epoch 9] train_loss: 0.304  test_accuracy: 0.954
9
train loss: 100%[**************************************************->]0.2797[epoch 10] train_loss: 0.298  test_accuracy: 0.952
10
train loss: 100%[**************************************************->]0.2785[epoch 11] train_loss: 0.303  test_accuracy: 0.951
11
train loss: 100%[**************************************************->]0.2421[epoch 12] train_loss: 0.291  test_accuracy: 0.955
12
train loss: 100%[**************************************************->]0.3261[epoch 13] train_loss: 0.289  test_accuracy: 0.950
13
train loss: 100%[**************************************************->]0.5185[epoch 14] train_loss: 0.291  test_accuracy: 0.952
14
train loss: 100%[**************************************************->]0.2864[epoch 15] train_loss: 0.283  test_accuracy: 0.953
15
train loss: 100%[**************************************************->]0.3263[epoch 16] train_loss: 0.278  test_accuracy: 0.935
16
train loss: 100%[**************************************************->]0.4107[epoch 17] train_loss: 0.288  test_accuracy: 0.950
17
train loss: 100%[**************************************************->]0.4473[epoch 18] train_loss: 0.277  test_accuracy: 0.951
18
train loss: 100%[**************************************************->]0.2263[epoch 19] train_loss: 0.276  test_accuracy: 0.952
19
train loss: 100%[**************************************************->]0.4685[epoch 20] train_loss: 0.275  test_accuracy: 0.952

0
train loss: 100%[**************************************************->]0.2044[epoch 1] train_loss: 0.280  test_accuracy: 0.952
1
train loss: 100%[**************************************************->]0.2819[epoch 2] train_loss: 0.254  test_accuracy: 0.952
2
train loss: 100%[**************************************************->]0.3506[epoch 3] train_loss: 0.252  test_accuracy: 0.957
3
train loss: 100%[**************************************************->]0.2673[epoch 4] train_loss: 0.242  test_accuracy: 0.958
4
train loss: 100%[**************************************************->]0.1955[epoch 5] train_loss: 0.238  test_accuracy: 0.958
5
train loss: 100%[**************************************************->]0.1683[epoch 6] train_loss: 0.241  test_accuracy: 0.959
6
train loss: 100%[**************************************************->]0.1992[epoch 7] train_loss: 0.235  test_accuracy: 0.957
7
train loss: 100%[**************************************************->]0.2159[epoch 8] train_loss: 0.231  test_accuracy: 0.955
8
train loss: 100%[**************************************************->]0.2265[epoch 9] train_loss: 0.232  test_accuracy: 0.959
9
train loss: 100%[**************************************************->]0.2372[epoch 10] train_loss: 0.234  test_accuracy: 0.960
10
train loss: 100%[**************************************************->]0.1721[epoch 11] train_loss: 0.228  test_accuracy: 0.960
11
train loss: 100%[**************************************************->]0.3537[epoch 12] train_loss: 0.229  test_accuracy: 0.957
12
train loss: 100%[**************************************************->]0.4825[epoch 13] train_loss: 0.225  test_accuracy: 0.955
13
train loss: 100%[**************************************************->]0.2957[epoch 14] train_loss: 0.227  test_accuracy: 0.960
14
train loss: 100%[**************************************************->]0.1950[epoch 15] train_loss: 0.223  test_accuracy: 0.961
15
train loss: 100%[**************************************************->]0.2097[epoch 16] train_loss: 0.217  test_accuracy: 0.961
16
train loss: 100%[**************************************************->]0.2386[epoch 17] train_loss: 0.223  test_accuracy: 0.960
17
train loss: 100%[**************************************************->]0.3696[epoch 18] train_loss: 0.221  test_accuracy: 0.961
18
train loss: 100%[**************************************************->]0.2363[epoch 19] train_loss: 0.217  test_accuracy: 0.961
19
train loss: 100%[**************************************************->]0.2499[epoch 20] train_loss: 0.221  test_accuracy: 0.961
20
train loss: 100%[**************************************************->]0.2369[epoch 21] train_loss: 0.215  test_accuracy: 0.962


```

