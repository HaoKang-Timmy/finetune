import argparse
import os
import random
import warnings
from utils import train, validate, adjust_learning_rate, save_checkpoint, prepare_dataloader, LiteResidualModule
from ofa.utils import replace_bn_with_gn, init_models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from ofa.model_zoo import proxylessnas_mobile
from ofa.utils.layers import LinearLayer
from torch.utils.tensorboard import SummaryWriter
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--a', '--arch', metavar='ARCH', default='mobilenet_v2',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('-type', '--dataset-type', default='Imagenet',
                    help='choose a dataset to train')
parser.add_argument('--gamma', default=0.9, type=float,
                    help='decay rate at scheduler')
parser.add_argument('--tensorboard', action='store_true',
                    help='set up a sesion at tensorboard')
parser.add_argument('--train-method', choices=[
                    'deep', 'low', 'finetune', 'bias', 'TinyTL-L', 'TinyTL-B', 'TinyTL-L+B', 'norm+last'], default='fintune', help='choose a training method')
parser.add_argument('--proxy', default=False, action='store_true')
best_acc1 = 0


def main():

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('choose seed training, may slow down training')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.proxy == False:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        if args.dataset_type == 'CUB200':
            model.classifier[-1] = nn.Linear(1280, 200)
        elif args.dataset_type == 'CIFAR10':
            model.classifier[-1] = nn.Linear(1280, 10)
        elif args.dataset_type == 'CIFAR100':
            model.classifier[-1] = nn.Linear(1280, 100)
    # still could just work on mobilenet_v2

    if args.proxy == True:
        classification_head = []
        model = proxylessnas_mobile(pretrained=True)
        if args.dataset_type == 'CUB200':
            model.classifier = LinearLayer(1280, 200, dropout_rate=0.2)
        elif args.dataset_type == 'CIFAR10':
            model.classifier = LinearLayer(1280, 10, dropout_rate=0.2)
        elif args.dataset_type == 'CIFAR100':
            model.classifier = LinearLayer(1280, 100, dropout_rate=0.2)
        classification_head.append(model.classifier)
        init_models(classification_head)
    if args.train_method == 'finetune':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.classifier.parameters(), args.lr,
                                     weight_decay=args.weight_decay)

    elif args.train_method == 'low':
        classifier_map = list(map(id, model.classifier.parameters()))
        low_map = list(map(id, model.features[-5:]))
        classifier_params = filter(lambda p: id(
            p) in classifier_map, model.parameters())
        low_params = filter(lambda p: id(p) in low_map, model.parameters())
        deep_params = filter(lambda p: id(
            p) not in low_map+classifier_map, model.parameters())
        optimizer = torch.optim.Adam([{'params': classifier_params}, {
            'params': low_params, 'lr': args.lr*0.6}, {'params': deep_params, 'lr': args.lr*0.4}], lr=args.lr, weight_decay=args.weight_decay)
    elif args.train_method == 'deep':
        for param in model.parameters():
            param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                         weight_decay=args.weight_decay)
    elif args.train_method == 'TinyTL-L':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        LiteResidualModule.insert_lite_residual(model)
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    elif args.train_method == 'TinyTL-L+B':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        LiteResidualModule.insert_lite_residual(model)
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    elif args.train_method == 'TinyTL-B':
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
        # l2sp_op =l2sp(model.parameters(), lr=args.lr*0.5)
    elif args.train_method == 'norm+last':
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'norm' in name:
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)
        replace_bn_with_gn(model, gn_channel_per_group=8)
    if not torch.cuda.is_available():
        print('using CPU')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])

        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        if args.tensorboard:
            print('create tensorboard session')
            writer = SummaryWriter()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    compose_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    compose_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_sampler, train_loader, val_loader = prepare_dataloader(
        normalize, compose_train, compose_val, args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        acc1_train, loss_train = train(train_loader, model, criterion,
                                       optimizer, epoch, args, ngpus_per_node)
        adjust_learning_rate(scheduler)
        acc1, loss_val = validate(
            val_loader, model, criterion, args, ngpus_per_node)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
            if args.tensorboard:
                print(loss_train, acc1_train, loss_val, acc1)
                writer.add_scalar('loss/train', loss_train, epoch)
                writer.add_scalar('acc/train', acc1_train, epoch)
                writer.add_scalar('loss/val', loss_val, epoch)
                writer.add_scalar('acc/val', acc1, epoch)
                train_loss_save = './log/tune_last.txt'
                file_save1 = open(train_loss_save, mode='a')
                file_save1.write('\n'+'step:'+str(epoch)+'  loss_train:'+str(loss_train)+'  acc1_train:'+str(
                    acc1_train.item())+'  loss_val:'+str(loss_val)+'  acc1_val:'+str(acc1.item()))
                print(scheduler.get_last_lr())
                file_save1.close()


if __name__ == '__main__':
    main()
