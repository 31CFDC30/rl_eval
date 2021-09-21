"""
这里定义一些常用的函数
作者: Mashun
"""
import torch

from builtins import tuple


def func_n2t(sample: tuple, device):
    sample = [torch.from_numpy(d).float().to(device) for d in sample]

    return sample


def func_t2n(sample: tuple):
    sample = [d.cpu().numpy() for d in sample]
    return sample


def func_to(sample, device):

    sample = [d.to(device) for d in sample]

    return sample

