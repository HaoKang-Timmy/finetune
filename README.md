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
   | 20     | 0.918    |


   

2. resnet34

   | Epochs | Accuracy |
   | ------ | -------- |
   | 20     | 0.932    |

3. vgg13

   | Epochs | Accuracy |
   | ------ | -------- |
   | 20     | 0.939    |


4. Densenet121

| Epochs | Accuracy |
| ------ | -------- |
| 20     | 0.922    |


# some experience

1. When batch_size is small, accuracy changes rapidly(up and down)
2. should manually change learning rate sometimes
3. could try big rate when acc does not change often
4. # another resources(trained 50 epochs but not fintuned)
![image](https://user-images.githubusercontent.com/60107867/139537346-7ce9b714-812f-4cd6-9a03-d3219cccef90.png)
