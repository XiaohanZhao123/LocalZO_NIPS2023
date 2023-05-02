from typing import Any

import torch
from torch.autograd import Function
from spconv.pytorch import SparseConvTensor
from utils import generate_sparse_input
from spconv import pytorch as spconv
from typing import Tuple, Optional


import torch

class SNNTest(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, batch_size) -> Any:
        print('shit')
        return values * 2, indices, batch_size


    @staticmethod
    def backward(ctx, grad_values, grad_indices, grad_batch_size) -> Tuple[Optional[torch.Tensor], ...]:
        print('shitter')
        return grad_values * 2, None, None


class SNNLayer(torch.nn.Module):

    def forward(self, input: SparseConvTensor):
        values = input.features
        indices = input.indices
        batch_size = input.batch_size
        values, indices, batch_size = SNNTest.apply(values, indices, batch_size)
        return SparseConvTensor(values, indices, input.spatial_shape, batch_size)


def snn_test():
    input = generate_sparse_input((16, 25, 25, 3), 0.9).cuda()
    input.requires_grad = True
    input = SparseConvTensor.from_dense(input)
    snn_layer = SNNLayer().cuda()
    output = snn_layer(input)
    layer = spconv.SparseConv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        stride=1,
        padding=1,
        algo=spconv.ConvAlgo.Native
    ).cuda()
    output = layer(output)
    output.dense().sum().backward()
    print(input.features.grad)


if __name__ == '__main__':
    snn_test()
