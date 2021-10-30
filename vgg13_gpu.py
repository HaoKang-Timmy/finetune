import torch
from torchvision import transforms
import torchvision
from  torch.utils import data
import torch.nn as nn
from torch import optim
batch_size = 128
data_transform = {
    
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
# load train data
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=data_transform["train"])
trainloader = data.DataLoader(dataset=trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
# load test data
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=False,
                                       transform=data_transform["val"])
testloader = data.DataLoader(dataset=testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
net = torchvision.models.vgg13(pretrained=True)
print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 # net into cuda
print(device)

for param in net.parameters():
    param.requires_grad = False
net.classifier[-1] = nn.Linear(4096,10)
for param in net.classifier.parameters():
    param.requires_grad = True
net.to(device)


# define ooptimizer and loss function
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# training
best_acc = 0.0
save_path = './vgg11.pth'
for epoch in range(20):
    if epoch>10 and flag ==0:
        lr = 0.0001
        fc_param = list(map(id, net.classifier[-1].parameters()))
        other_param = filter(lambda p: id(p) not in fc_param, net.parameters())
        optimizer = torch.optim.Adam(
            [{'params': fc_param}, {'params': other_param, 'lr': lr*0.1}])
        flag = 1
    print(epoch)
    net.train()
    running_loss = 0.0
    for step, data in enumerate(trainloader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        rate = (step+1)/len(trainloader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print(
            "\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")

    net.eval()
    acc = 0.0
    test_num = 0
    with torch.no_grad():
        for val_data in testloader:
            val_images, val_labels = val_data
            outputs = net(val_images)  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels).sum().item()
            test_num += 1
        val_accurate = acc / (test_num * batch_size)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
