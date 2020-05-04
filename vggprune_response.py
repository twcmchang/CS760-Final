import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        if 'cfg' not in checkpoint.keys():
            model = vgg(dataset=args.dataset, depth=args.depth)
        else:
            model = vgg(dataset=args.dataset, depth=args.depth,
                        cfg=checkpoint['cfg'])
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if args.cuda:
    model.cuda()

print(model)

def test(model):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        if args.dataset == 'cifar10':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        elif args.dataset == 'cifar100':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        else:
            raise ValueError("No valid dataset is given.")
        model.eval()
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        #     correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        return correct.item() / len(test_loader.dataset)

total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)

original_model = model
percent = np.concatenate((np.arange(0.05, 0.7, 0.05), np.arange(0.7, 0.8, 0.01), np.arange(0.8, 1,0.05)))

for perc in percent:
    model = original_model
    thre_index = int(total * perc)
    thre = y[thre_index]
    thre = thre.to(device='cuda')

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float()
            # avoid pruning to zero channels
            if torch.sum(mask) == 0:
                mask[0] = 1.0
            mask = mask.cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                # format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total

    # print('Pre-processing Successful!')

    # simple test model after Pre-processing prune (simple set BN scales to zeros)

    acc = test(model)
    print("{:.4f}, {:.4f}".format(perc, acc))


