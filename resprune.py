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
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.model:
    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model)
        if 'cfg' not in checkpoint.keys():
            print('not cfg')
            model = resnet(depth=args.depth, dataset=args.dataset)
        else:
            model = resnet(dataset=args.dataset,
                           depth=args.depth, cfg=checkpoint['cfg'])
            print(checkpoint['cfg'])
        print("=> loading checkpoint '{}'".format(args.model))
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if args.cuda:
    model.cuda()

total = 0
modules = list(model.modules())
for k, m in enumerate(modules):
    if isinstance(m, nn.BatchNorm2d):
        if isinstance(modules[k+1], channel_selection):
            indexes = modules[k+1].indexes.data.clone()
            mask = np.asarray(indexes.cpu().numpy())
            total += int(np.sum(mask))
        else:
            total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for k, m in enumerate(modules):
    if isinstance(m, nn.BatchNorm2d):
        if isinstance(modules[k+1], channel_selection):
            indexes = modules[k+1].indexes.data.clone()
            mask = np.asarray(indexes.cpu().numpy())
            size = int(np.sum(mask))
            idx1 = np.squeeze(np.argwhere(mask))
            bn[index:(index+size)] = m.weight.data[idx1.tolist()].abs().clone()
        else:
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

# find the threshold
y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]
print("threshold=", thre.item())
thre = thre.to(device='cuda')


pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(modules):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()

        if isinstance(modules[k+1], channel_selection):
            indexes = modules[k+1].indexes.data.clone()
            weight_copy.mul_(indexes)

        mask = weight_copy.gt(thre).float()
        # avoid pruning to zero channels
        if torch.sum(mask) == 0:
            mask[torch.argmax(weight_copy)] = 1.0
        mask = mask.cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        if isinstance(modules[k+1], channel_selection):
            print('layer index: {:d} \t total channel: {:d}/{:d} \t remaining channel: {:d}'.
                  format(k, int(torch.sum(indexes)), mask.shape[0], int(torch.sum(mask))))
        else:
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
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

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct.item() / float(len(test_loader.dataset))


# acc = test(model)

print("Cfg:")
print(cfg)

newmodel = resnet(depth=args.depth, dataset=args.dataset, cfg=cfg)
if args.cuda:
    newmodel.cuda()

old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # We need to set the channel selection layer.
            m2 = new_modules[layer_id + 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0
        else:
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()

        print(layer_id, "BN: ", m1.weight.data.shape)
        print(idx0)
        print(idx1)
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            print(layer_id, "Conv1: ", m1.weight.shape)
            continue
        if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
            # This convers the convolutions in the residual block.
            # The convolutions are either after the channel selection layer or after the batch normalization layer.
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(
                np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            print("idx0: ", idx0)
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            # If the current convolution is not the last convolution in the residual block, then we can change the
            # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
            if conv_count % 3 != 1:
                w1 = w1[idx1.tolist(), :, :, :].clone()
                print("idx1: ", idx1)
            m1.weight.data = w1.clone()
            print(layer_id, "Conv2: ", m1.weight.shape)
            continue

        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.
        m1.weight.data = m0.weight.data.clone()
        print(layer_id, "Conv3: ", m1.weight.shape)
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()},
           os.path.join(args.save, 'pruned.pth.tar'))

print(newmodel)
acc = test(newmodel)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))
