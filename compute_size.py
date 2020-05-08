import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from utils import print_model_param_nums, print_model_param_flops


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.arch == "vgg":
    model_type = vgg
elif args.arch == "resnet":
    model_type = resnet
elif args.arch == "densenet":
    model_type = densenet
else:
    print("=> no arch {} found.".format(args.arch))
    raise SystemExit(0)

if args.model:
    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model)
        if 'cfg' not in checkpoint.keys():
            model = model_type(depth=args.depth, dataset=args.dataset)
        else:
            model = model_type(dataset=args.dataset,
                           depth=args.depth, cfg=checkpoint['cfg'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        raise SystemExit(0)

if args.cuda:
    model.cuda()

num_paras = print_model_param_nums(model)
num_flops = print_model_param_flops(model)
print(num_paras, num_flops)
