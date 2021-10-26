# Finetune_test
I have tested four models for cifar10, using adam optimizer\
1. vgg13
2. resnet34
3. alexnet
4. densenet121
each have trained at least 3 epochs
For vgg13 and alexnet,since their classifier are several layers but not just a fc,
there results are better.
For the other two, it is kind of hard to train them.
# results
1. alexnet

   | Epochs | Accuracy |
   | ------ | -------- |
   | 1      | 0.788    |
   | 2      | 0.814    |
   | 3      | 0.835    |
   | 4      | 0.841    |

   

2. resnet34

   | Epochs | Accuracy |
   | ------ | -------- |
   | 1      | 0.714    |
   | 2      | 0.724    |
   | 3      | 0.732    |
   | 4      | 0.737    |
   | 5      | 0.742    |
   | 6      | 0.821    |
   | 7      | 0.837    |

3. vgg13

   | Epochs | Accuracy |
   | ------ | -------- |
   | 1      | 0.821    |
   | 2      | 0.837    |
   | 3      | 0.848    |
   | 4      | 0.857    |

4. Densenet121

| Epochs | Accuracy |
| ------ | -------- |
| 1      | 0.596    |
| 2      | 0.671    |
| 3      | 0.698    |
| 4      | 0.714    |
| 5      | 0.723    |
|        |          |

# some experience

1. When batch_size is small, accuracy changes rapidly(up and down)
2. should manually change learning rate sometimes
3. could try big rate when acc does not change often
