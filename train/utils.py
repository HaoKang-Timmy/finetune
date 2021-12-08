import time
import torch
import shutil
from vision.vision_class import AverageMeter, ProgressMeter
from dataset.dataset_collection import DatasetCollection
import torchvision.transforms as transforms
import torch.utils.data.distributed
from torch.optim.optimizer import Optimizer
from torch.autograd import Function as F
from typing import List, Optional
from torch import Tensor
import math
import torch.nn as nn
from ofa.utils import get_same_padding, make_divisible, build_activation, init_models
from collections import OrderedDict
from ofa.imagenet_classification.networks import ProxylessNASNets
from ofa.utils.layers import ZeroLayer


def train(train_loader, model, criterion, optimizer, epoch, args, ngpus_per_node, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     writer.add_scalar('Loss/train', losses.val, i)
        #     writer.add_scalar('acc/train', top1.val, i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args, ngpus_per_node, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            # and args.rank % ngpus_per_node == 0):
            #     writer.add_scalar('Loss/val', losses.val, i)
            #     writer.add_scalar('acc/val', top1.val, i)
            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def prepare_dataloader(normalize, compose_train, compose_val, args):
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
    dataset_collection = DatasetCollection(
        args.dataset_type, args.data, compose_train, compose_val)
    train_dataset, val_dataset = dataset_collection.init()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return train_sampler, train_loader, val_loader


def adjust_learning_rate(scheduler):
    scheduler.step()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class l2sp(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.oldparam = params
        super(l2sp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(l2sp, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)
                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
            print(params_with_grad)
            l2sp_adam(params_with_grad,
                      grads,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      state_steps,
                      amsgrad=group['amsgrad'],
                      beta1=beta1,
                      beta2=beta2,
                      lr=group['lr'],
                      weight_decay=group['weight_decay'],
                      eps=group['eps'])
        return loss


def l2sp_adam(params: List[Tensor],
              grads: List[Tensor],
              exp_avgs: List[Tensor],
              exp_avg_sqs: List[Tensor],
              max_exp_avg_sqs: List[Tensor],
              state_steps: List[int],
              *,
              amsgrad: bool,
              beta1: float,
              beta2: float,
              lr: float,
              weight_decay: float,
              eps: float, l2sp=0, old_param):
    for i, param in enumerate(params):
        old_param = old_param[i]
        grad = grads[i]
        step = state_steps[i]
        grad = grad.add(param, alpha=weight_decay)
        grad = grad.add(param, alpha=-old_param)

    pass


class LiteResidualModule(nn.Module):

    def __init__(self, main_branch, in_channels, out_channels,
                 expand=1.0, kernel_size=3, act_func='relu', n_groups=2,
                 downsample_ratio=2, upsample_type='bilinear', stride=1):
        super(LiteResidualModule, self).__init__()
        self.main_branch = main_branch
        self.lite_residual_config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'expand': expand,
            'kernel_size': kernel_size,
            'act_func': act_func,
            'n_groups': n_groups,
            'downsample_ratio': downsample_ratio,
            'upsample_type': upsample_type,
            'stride': stride,
        }
        kernel_size = 1 if downsample_ratio is None else kernel_size
        padding = get_same_padding(kernel_size)
        pooling = nn.AvgPool2d(downsample_ratio, downsample_ratio, 0)
        num_mid = make_divisible(int(in_channels * expand), divisor=8)
        self.lite_residual = nn.Sequential(OrderedDict({
            'pooling': pooling,
            'conv1': nn.Conv2d(in_channels, num_mid, kernel_size, stride, padding, groups=n_groups, bias=False),
            'bn1': nn.BatchNorm2d(num_mid),
            'act': build_activation(act_func),
            'conv2': nn.Conv2d(num_mid, out_channels, 1, 1, 0, bias=False),
            'final_bn': nn.BatchNorm2d(out_channels),
        }))
        print(self.lite_residual)
        init_models(self.lite_residual)
        self.lite_residual.final_bn.weight.data.zero_()

    def forward(self, x):
        main_x = self.main_branch(x)
        lite_residual_x = self.lite_residual(x)
        if self.lite_residual_config['downsample_ratio'] is not None:
            lite_residual_x = F.upsample(lite_residual_x, main_x.shape[2:],
                                         mode=self.lite_residual_config['upsample_type'])
        return main_x + lite_residual_x

    @staticmethod
    def insert_lite_residual(net, downsample_ratio=2, upsample_type='bilinear',
                             expand=1.0, max_kernel_size=5, act_func='relu', n_groups=2,
                             **kwargs):
        if isinstance(net, ProxylessNASNets):
            bn_param = net.get_bn_param()
            max_resolution = 128
            stride_stages = [2, 2, 2, 1, 2, 1]
            for block_index_list, stride in zip(net.grouped_block_index, stride_stages):
                for i, idx in enumerate(block_index_list):
                    block = net.blocks[idx].conv
                    if isinstance(block, ZeroLayer):
                        continue
                    s = stride if i == 0 else 1
                    block_downsample_ratio = downsample_ratio
                    block_resolution = max(
                        1, max_resolution // block_downsample_ratio)
                    max_resolution //= s

                    kernel_size = max_kernel_size
                    if block_resolution == 1:
                        kernel_size = 1
                        block_downsample_ratio = None
                    else:
                        while block_resolution < kernel_size:
                            kernel_size -= 2
                    net.blocks[idx].conv = LiteResidualModule(
                        block, block.in_channels, block.out_channels, expand=expand, kernel_size=kernel_size,
                        act_func=act_func, n_groups=n_groups, downsample_ratio=block_downsample_ratio,
                        upsample_type=upsample_type, stride=s,
                    )

            net.set_bn_param(**bn_param)
        else:
            for i in range(1, 18):
                print(i)
                print(net.features[i])
                if i == 1:
                    net.features[i] = LiteResidualModule(net.features[i], in_channels=net.features[i].conv[0][0].in_channels, out_channels=net.features[i].conv[1].out_channels, expand=expand, kernel_size=3,
                                                         act_func=act_func, n_groups=n_groups, downsample_ratio=downsample_ratio,
                                                         upsample_type=upsample_type, stride=net.features[i].conv[0][0].stride[1],)
                else:
                    net.features[i] = LiteResidualModule(net.features[i], in_channels=net.features[i].conv[0][0].in_channels, out_channels=net.features[i].conv[2].out_channels, expand=expand, kernel_size=3,
                                                         act_func=act_func, n_groups=n_groups, downsample_ratio=downsample_ratio,
                                                         upsample_type=upsample_type, stride=net.features[i].conv[1][0].stride[1],)
