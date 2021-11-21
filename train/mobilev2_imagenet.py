from torchvision import transforms
from dataset import DatasetCollection
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.utils.data
import torch.nn as nn
type = 'Imagenet'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--device',default='cpu')
parser.add_argument('--batchsize',default=16,type = int)
parser.add_argument('--pretrain',default= True)
parser.add_argument('-f','--fintune_mode',default = 'low')
parser.add_argument('-l','--learning_rate',default= 0.001)
args = parser.parse_args()
dr = args.data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
compose_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
compose_val = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
mydataset = DatasetCollection(type,dr,compose_train,compose_val)
train,val = mydataset.init()
train_loader = torch.utils.data.DataLoader(
    train, batch_size=args.batchsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val,
    batch_size=args.batchsize, shuffle=False)
device = torch.device(args.device)
net = torchvision.models.mobilenet_v2(pretrained=args.pretrain)
net.classifier[-1] = nn.Linear(1280,1000)
if args.fintune_mode == 'low':
    for param in net.parameters():
        param.requires_grad = False
    for param in net.classifier.parameters():
        param.requires_grad = True
if args.fintune_mode == 'deep':
    for param in net.parameters():
        param.requires_grad = True
if args.device!= 'cpu':
    net = nn.DataParallel(net).to(device)
loss_function = nn.CrossEntropyLoss()
save_path = '/path'
def train():
    best_acc = 0.68
    #optimizer = torch.optim.Adam([{'params':classifier_params},{'params':low_params,'lr':lr_init*0.4},{'params':deep_params,'lr':lr_init*0.1}],lr=lr_init)
    optimizer = torch.optim.Adam(net.module.parameters(),lr = args.learning_rate,weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.8)
    for i in range(20):
        #optimizer = torch.optim.Adam(net.module.parameters(),lr = lr_init,weight_decay=weight_decay)
        #optimizer = torch.optim.Adam([{'params':classifier_params},{'params':low_params,'lr':lr_init*0.6},{'params':deep_params,'lr':lr_init*0.4}],lr=lr_init)
        net.train()
        running_loss = 0.0
        
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()
            images = images.to(device)
            logits = net(images)
            labels = torch.tensor(labels).to(device)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            rate = (step+1)/len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print(
                "\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        scheduler.step()
        if i in [0,2,4,6,8,10,12,14,16,18]:
            print(i)
            net.eval()
            acc = 0.0
            test_num = 0.0
            with torch.no_grad():
                for val_data in val_loader:
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