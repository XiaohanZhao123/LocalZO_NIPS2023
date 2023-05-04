import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
import torch
import torchvision
import functools
from experiments.models.examples import SpconvNet, ExampleNetLocalZOOnce
from utils import train_and_profile_spconv

# dataloader arguments
batch_size = 128
data_path = '/home/zxh/data'


def get_dataset():
    sensor_size = tonic.datasets.NMNIST.sensor_size
    # remove isolated events
    # sum a period of events into a frame
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])
    trainset = tonic.datasets.NMNIST(save_to=data_path, train=True, transform=frame_transform)

    transform = tonic.transforms.Compose([
        torch.from_numpy,
        torchvision.transforms.RandomRotation([-10, 10]),
    ])

    cached_trainset = DiskCachedDataset(trainset, cache_path='./data/cache', transform=transform)

    # here, we need to pad the frame to the same length by using collate_fn
    # we then make time-first dataset by setting batch_first=False
    batch_size = 128
    train_loader = DataLoader(cached_trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=tonic.collation.PadTensors(batch_first=False))


    # check the shape of input
    next(iter(train_loader))[0].shape
    return train_loader


def grad_sparsity():
    spconv_net = SpconvNet(beta=0.5, u_th=0.8, batch_size=batch_size).cuda()
    spconvzo_net = ExampleNetLocalZOOnce(batch_size=batch_size, beta=0.5, u_th=0.8).cuda()

    train_loader = get_dataset()
    train_and_profile_spconv(spconv_net, train_loader, SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2),
                             SF.ce_rate_loss)
    train_and_profile_spconv(spconvzo_net, train_loader, SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2),
                             SF.ce_rate_loss)


if __name__ == '__main__':
    grad_sparsity()
