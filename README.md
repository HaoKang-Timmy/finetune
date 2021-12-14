---
typora-copy-images-to: ./pic
---

# 1 Batchsize vs accuracy

## 1.1 Theory

There are mainly two reasons that constrain batch size and accuracy. As for the first reason, our model is trying to solve a non-convex optimization problem that has a lot of local optima. If we choose a large batch size, we could get into local optima easily[1]. Small batch size's loss decays not as steady as the large batch size. Therefore, choosing a small batch size has a smaller possibility to stay at a local optimum.

![image-20211214141236539](./pic/image-20211214141236539.png) 

Another reason is that when choosing SGD as our optimizer(if we choose others, we have similar results, too), the backward step of large batch size is not equal to seldom backward steps of small batch size.

If we have a minibatch size of n pictures in image classification problems, and a large batch is n times of a minibatch. We could see that the loss functions are different((3) and (4)). We assume η is the learning rate.

![image-20211214143631749](./pic/image-20211214143631749.png)

If we assume that $w_t ≈ w_{t+k} $ , which means that after each batch's backward parameters change a little, (3) and (4) are the same.

## 1.2 Main Results

### 1.2.1 Settings

| setting            | value                                                        |
| ------------------ | ------------------------------------------------------------ |
| Pretrained dataset | Imagenet                                                     |
| Dataset            | CIFAR10                                                      |
| Epochs             | 50                                                           |
| Optimizer          | SGD                                                          |
| Momentum           | 0.9                                                          |
| Lr(initial)        | Scaling learning rate with batch size                        |
| Scheduler          | CosineAnnealingLR                                            |
| Batch size         | 8 , 256                                                      |
| Weight decay       | 1e-4                                                         |
| Distributed        | No                                                           |
| Backbone           | MobileNetV2                                                  |
| Finetune strategy  | Random Weights（MobileNetV2)/hanlab pretrained(ProxylessNAS-Mobile) |

 



# 2 The reason that slow down the speed of training

## 1.1 Theory

One reason is that `torch.backends.cudnn.deterministic.` This will make sure that if settings are the same, the training parameters of models are the same in each step. However, this will decrease training speed. Only manually set random seed will not cause decreases.

The other reason is that workers of data loader. Data are stored at disks. CPU manages them and sends them to GPUs. If bandwidth is small, GPUs calculate fast and wait for data. This is the main reason that slows down the speed.

## 1.2 Main Results

Batch size = 256

Backbone = MobileNetV2

Picture size = 3\*224\*224

| CPU workers | time per batch per GPU |
| ----------- | ---------------------- |
| 4           | 4.9s                   |
| 8           |                        |
| 16          | 1.8s                   |



# Reference

[1] Keskar N S, Mudigere D, Nocedal J, et al. On large-batch training for deep learning: Generalization gap and sharp minima[J]. arXiv preprint arXiv:1609.04836, 2016.

