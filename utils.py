"""Common functions you may find useful in your implementation."""
import os
import json
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


EPS = 1e-7


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_zero_grads(params):
    for p in params:
        if p.grad is not None:
            assert_eq(p.grad.data.sum(), 0)


def assert_frozen(module):
    for p in module.parameters():
        assert not p.requires_grad


def weights_init(m):
    """custom weights initialization"""
    classtype = m.__class__
    if classtype == nn.Linear or classtype == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
    elif classtype == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % classtype)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def count_output_size(input_shape, module):
    fake_input = Variable(torch.FloatTensor(*input_shape), volatile=True)
    output_size = module.forward(fake_input).view(-1).size()[0]
    return output_size


def one_hot(x, n):
    assert x.dim() == 2
    one_hot_x = torch.zeros(x.size(0), n).cuda()
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x


def large_randint():
    return random.randint(int(1e5), int(1e6))


def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(large_randint())
    torch.manual_seed(large_randint())
    torch.cuda.manual_seed(large_randint())


class Config(object):
    def __init__(self, attrs):
        self.__dict__.update(attrs)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            attrs = json.load(f)
        return cls(attrs)

    def dump(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print('Results will be stored in:', dirname)

        with open(filename, 'w') as f:
            json.dump(vars(self), f, sort_keys=True, indent=2)
            f.write('\n')

    def __repr__(self):
        return json.dumps(vars(self), sort_keys=True, indent=2)
