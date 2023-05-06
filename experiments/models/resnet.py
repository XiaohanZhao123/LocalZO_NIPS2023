import functools

import snntorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv import pytorch as spconv
from LocalZO.conv_models.neurons import LeakyPlain, LeakyZOPlainOnce
from snntorch import surrogate
from spconv.pytorch.functional import sparse_add
from snntorch import utils

spike_grad = surrogate.atan()


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, conv_cls, batchnorm_cls, snn_cls):
        """
        basic blocks of resnet, which is used to build the resnet network

        Args:
            in_channel: the input channel of the block
            out_channel:  the output channel of the block
            stride: convolution stride
            conv_cls: the class of convolution layer
            snn_cls: the class of spiking neuron
            batchnorm_cls:the class of batch normalization layer
        """
        super(ResBlock, self).__init__()

        self.left = nn.Sequential(
            conv_cls(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                     bias=False),
            batchnorm_cls(out_channels),
            snn_cls(),
            conv_cls(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                     bias=False),
            batchnorm_cls(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # if the input channel is not equal to the output channel, we need to use 1*1 convolution to change the
            # channel
            self.shortcut = nn.Sequential(
                conv_cls(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                batchnorm_cls(out_channels)
            )
        self.snn_layer = snn_cls()

    def forward(self, x: spconv.SparseConvTensor):
        out = self.left(x)
        if isinstance(x, spconv.SparseConvTensor):
            out = out.dense() + self.shortcut(x).dense()
            out = self.snn_layer(out)
            out = spconv.SparseConvTensor.from_dense(out.transpose(1, 3))
        else:
            out += self.shortcut(x)
            out = self.snn_layer(out)

        return out


class ResNet18Spconv(nn.Module):
    def __init__(self, snn_cls, u_th, beta, batch_size, num_class=10):
        """
        ResNet18 network

        Args:
            conv_cls: the class of convolution layer
            batchnorm_cls: the class of batch normalization layer
            snn_cls: the class of spiking neuron
            neuron_params: the parameters of the spiking neuron
            num_class: the number of classes of the classification task
        """
        super(ResNet18Spconv, self).__init__()

        self.batch_size = batch_size
        self.num_class = num_class

        algo = spconv.ConvAlgo.Native
        conv_cls = functools.partial(spconv.SparseConv2d, algo=algo, )
        avg_pool_cls = functools.partial(spconv.SparseAvgPool2d, algo=algo, )
        batchnorm_cls = spconv.SparseBatchNorm
        assert snn_cls in [LeakyPlain, LeakyZOPlainOnce], 'snn_cls must be LeakyPlain or LeakyZOPlainOnce'
        snn_cls = functools.partial(snn_cls, u_th=u_th, beta=beta, batch_size=batch_size, )
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            conv_cls(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            spconv.SparseAvgPool2d(kernel_size=3, stride=2, padding=1, algo=algo, ),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size)
        )
        self.layer1 = self.make_layer(64, 2, 1, conv_cls, batchnorm_cls, snn_cls, )
        self.layer2 = self.make_layer(128, 2, 2, conv_cls, batchnorm_cls, snn_cls, )
        self.layer3 = self.make_layer(256, 2, 2, conv_cls, batchnorm_cls, snn_cls, )
        self.layer4 = self.make_layer(512, 2, 2, conv_cls, batchnorm_cls, snn_cls, )
        self.avg_pool = avg_pool_cls(kernel_size=2)
        self.to_dense = spconv.ToDense()
        self.fc = nn.Linear(512, num_class)
        self.snn_out = LeakyPlain(u_th, beta, batch_size)

    def make_layer(self, out_channel, num_blocks, stride, conv_cls, batchnorm_cls, snn_cls):
        """
        make the layers of the resnet

        Args:
            out_channel: the output channel of the layer
            num_blocks: the number of blocks in the layer
            stride: convolution stride
            conv_cls: the class of convolution layer
            batchnorm_cls: the class of batch normalization layer
            snn_cls: the class of spiking neuron

        Returns: the layer of the resnet

        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channel, out_channel, stride, conv_cls, batchnorm_cls, snn_cls))
            self.in_channel = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, *x.shape[2:]).transpose(1, 3)  # move channel to the last one
        x = spconv.SparseConvTensor.from_dense(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self.to_dense(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        x = self.snn_out(x)
        x = x.view(-1, self.batch_size, self.num_class)

        return x


class ResNet18Snntorch(nn.Module):

    def __init__(self, u_th, beta, num_class):
        """
        implement the resnet18 network with snntorch for spike neuron and torch.nn for convs

        Args:
            u_th: threshold of the spiking neuron
            beta: drop rate of the spiking neuron
            num_class: the number of classes of the classification task
        """
        super(ResNet18Snntorch, self).__init__()
        self.num_class = num_class
        snn_cls = functools.partial(snntorch.Leaky, beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True)
        conv_cls = nn.Conv2d
        batchnorm_cls = nn.BatchNorm2d
        avg_pool = nn.AvgPool2d(kernel_size=2)
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            conv_cls(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ),
            batchnorm_cls(64),
            snn_cls(),
        )
        self.layer1 = self.make_layer(64, 2, 1, conv_cls, batchnorm_cls, snn_cls, )
        self.layer2 = self.make_layer(128, 2, 2, conv_cls, batchnorm_cls, snn_cls, )
        self.layer3 = self.make_layer(256, 2, 2, conv_cls, batchnorm_cls, snn_cls, )
        self.layer4 = self.make_layer(512, 2, 2, conv_cls, batchnorm_cls, snn_cls, )
        self.avg_pool = avg_pool
        self.fc = nn.Linear(512, num_class)
        self.snn_out = snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, output=True, init_hidden=True)

    def make_layer(self, out_channel, num_blocks, stride, conv_cls, batchnorm_cls, snn_cls):
        """
        make the layers of the resnet

        Args:
            out_channel: the output channel of the layer
            num_blocks: the number of blocks in the layer
            stride: convolution stride
            conv_cls: the class of convolution layer
            batchnorm_cls: the class of batch normalization layer
            snn_cls: the class of spiking neuron

        Returns: the layer of the resnet

        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channel, out_channel, stride, conv_cls, batchnorm_cls, snn_cls))
            self.in_channel = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.snn_out(x)

        return x


def get_models(batch_size, u_th, beta, num_class):
    torchnet = ResNet18Snntorch(u_th=u_th, beta=beta, num_class=num_class).cuda()
    spconvnet = ResNet18Spconv(snn_cls=LeakyZOPlainOnce, u_th=u_th, beta=beta, batch_size=batch_size,
                               num_class=num_class).cuda()
    return spconvnet, torchnet
