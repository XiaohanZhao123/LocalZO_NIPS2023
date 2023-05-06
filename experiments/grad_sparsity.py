import os

import snntorch.functional

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
import hydra
from hydra.utils import call
from dataset.utils import repeat

torch.manual_seed(0)

# dataloader arguments
batch_size = 128
data_path = '/home/zxh/data'
loss_fn = snntorch.functional.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

from utils import forward_snntorch

@hydra.main(config_path='config', config_name='defaults', version_base=None)
def main(cfg):
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    print(f'testing on {model_name}, {dataset_name}')
    print(f'model setup: u_th:{cfg.model.u_th}, beta:{cfg.model.beta}')
    lr = cfg.optimizer.lr
    constant_encoding = cfg.dataset.constant_encoding
    num_steps = cfg.dataset.num_step if constant_encoding else None
    print(f'num_steps:{num_steps}')
    train_loader = call(cfg.dataset.get_dataset)
    spconv_model, snntorch_model = call(cfg.model.get_models)
    inputs, labels = next(iter(train_loader))

    if constant_encoding:
        inputs = repeat(inputs, num_steps)
        inputs = inputs.cuda()
        labels = labels.cuda()





if __name__ == '__main__':
    main()