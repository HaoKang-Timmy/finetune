# Report of Fintune

# Menu

- [Report of Fintune](#report-of-fintune)
- [Menu](#menu)
- [Settings](#settings)
- [Results(best fintune strategy)](#results-best-fintune-strategy-)
  * [table](#table)
  * [curve](#curve)
    + [CIFAR10](#cifar10)
    + [CUB200](#cub200)
    + [Imagenet](#imagenet)
- [Strategy Compare](#strategy-compare)
  * [Standard Fine-tuning vs separate lr](#standard-fine-tuning-vs-separate-lr)
    + [result](#result)
    + [CIFAR10](#cifar10-1)
    + [CUB200](#cub200-1)
  * [seperate lr vs Fine-tuning last-3](#seperate-lr-vs-fine-tuning-last-3)
    + [CIFAR10](#cifar10-2)
- [Train-from-scratch vs fintune](#train-from-scratch-vs-fintune)
  * [Training for same epochs](#training-for-same-epochs)
  * [training for more epochs](#training-for-more-epochs)
  * [Difference of two sets of parameters](#difference-of-two-sets-of-parameters)
    + [cos similarity of bias](#cos-similarity-of-bias)



# Settings

Arch: MobileNetV2

Dataset:CIFAR10,CUB200,CAR196,FOOD101,CIFAR100

# Results(best fintune strategy)

## table

| Dataset  | Val_acc%(top1) | Test_acc%(top1) |
| -------- | -------------- | --------------- |
| Cifar10  | 96.224         |                 |
| Car196   | 87.295         |                 |
| Food101  | 8.124          |                 |
| Cifar100 | 73.242         |                 |
| CUB200   | 78.141         |                 |
| Imagenet | 70.241         |                 |

## curve

### CIFAR10

X-axis: epoch

Y-axis: acc%

![image-20211125100333222](./pic/image-20211125100333222.png)

X-axis: epoch

Y-axis: loss

![image-20211125100355745](./pic/image-20211125100355745.png)

### CUB200

![image-20211125101410191](./pic/image-20211125101410191.png)

![image-20211125101527341](./pic/image-20211125101527341.png)

### Imagenet

![image-20211125101612803](./pic/image-20211125101612803.png)

![image-20211125101635876](./pic/image-20211125101635876.png)

# Strategy Compare

Since Imagenet is too slow to train, I use CIFAR10 to implement these strategy.

## Standard Fine-tuning vs separate lr 

Standard Fine-tuning

```python
                optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                             weight_decay=args.weight_decay)
```

separate lr 

```python
            classifier_map = list(map(id, model.classifier.parameters()))
            low_map = list(map(id, model.features[-5:]))
            classifier_params = filter(lambda p: id(
                p) in classifier_map, model.parameters())
            low_params = filter(lambda p: id(p) in low_map, model.parameters())
            deep_params = filter(lambda p: id(
                p) not in low_map+classifier_map, model.parameters())
            optimizer = torch.optim.Adam([{'params': classifier_params}, {
                                         'params': low_params, 'lr': args.lr*0.6}, {'params': deep_params, 'lr': args.lr*0.4}], lr=args.lr)
```

### result

### CIFAR10

Blue curve:separate lr 

Orange curve:Standard Fine-tuning

![image-20211125112331630](./pic/image-20211125112331630.png)

![image-20211125112353758](./pic/image-20211125112353758.png)

### CUB200

Pink curve: separate lr 

Green curve: Standard Fine-tuning

![image-20211125113624240](./pic/image-20211125113624240.png)

![image-20211125113844084](./pic/image-20211125113844084.png)

It could be infer that using different lr at different layer, which is small lr for deep layers gets better results in validation datasets comparing to same lr for all layers.

Which is similar to this article,https://arxiv.org/pdf/1811.08737.pdf.



## seperate lr vs Fine-tuning last-3 

### CIFAR10

green curve:fFine-tuning last-3 

blue curve: seperate lr

![image-20211125201352556](./pic/image-20211125201352556.png)

# Train-from-scratch vs fintune

In this section, I choose the different layer with different lr fintune method shown above.

## Training for same epochs

red curve: train-from-scratch

Blue curve: fintune

![image-20211125131055325](./pic/image-20211125131055325.png)



Using same learning rate decay strategy(exp decay), we can not get similar results.

## training for more epochs

However, when I try to continue training, it seems to stop to grow.

![image-20211125143520267](./pic/image-20211125143520267.png)

![image-20211125143603418](./pic/image-20211125143603418.png)

Then I try to use these two sets of parameters to analyze the difference.

## Difference of two sets of parameters

### cos similarity of bias

![image-20211125200847772](./pic/image-20211125200847772.png)

However I still need a technique to analyse weight(tensors), it is to be done
