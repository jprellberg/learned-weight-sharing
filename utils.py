import argparse
import random
import string
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def forward(self, x):
        return x


def loop_iter(iter):
    while True:
        for item in iter:
            yield item


def grad_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def count_parameters(model, grad_only=True):
    return sum(p.numel() for p in model.parameters() if not grad_only or p.requires_grad)


def unique_string():
    return '{}.{}'.format(datetime.now().strftime('%Y%m%dT%H%M%SZ'),
                          ''.join(random.choice(string.ascii_uppercase) for _ in range(4)))


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def stratified_split(dataset, labels, train_size):
    from sklearn.model_selection import train_test_split
    x = list(range(len(labels)))
    train, test = train_test_split(x, train_size=train_size, stratify=labels, random_state=666)
    train = Subset(dataset, train)
    test = Subset(dataset, test)
    return train, test
