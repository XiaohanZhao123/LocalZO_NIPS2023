import torch
from torch import nn
from torch.autograd import Function
from spconv import pytorch as spconv
from typing import Tuple, Optional, Any
from .utils import generate_sparse_input
from .functional import PlainLIFFunction

class LeakyPlain(nn.Module):

    def __init__(self, u_th, beta, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.beta = beta
        self.u_th = u_th

    def forward(self, inputs: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        if isinstance(inputs, torch.Tensor):
            outputs = PlainLIFFunction.apply(inputs, self.batch_size, self.u_th, self.beta)
            return outputs

        # else
        inputs = inputs.dense()
        outputs = PlainLIFFunction.apply(inputs, self.batch_size, self.u_th, self.beta)
        return spconv.SparseConvTensor.from_dense(outputs.transpose(1, 3))


class DummyConvSNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = spconv.SparseConv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            algo=spconv.ConvAlgo.Native
        )
        self.ac1 = LeakyPlain(0.5, 0.5, 64)
        self.conv2 = spconv.SparseConv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            algo=spconv.ConvAlgo.Native
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        return x


if __name__ == '__main__':
    net = DummyConvSNN().cuda()
    inputs = generate_sparse_input((64 * 10, 25, 25, 3), 0.9).cuda()
    inputs.requires_grad = True
    inputs = spconv.SparseConvTensor.from_dense(inputs)
    output = net(inputs)
    output.dense().sum().backward()
    a = 10
