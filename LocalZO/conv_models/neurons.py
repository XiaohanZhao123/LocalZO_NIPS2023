import torch
from torch import nn
from torch.autograd import Function
from spconv import pytorch as spconv
from typing import Tuple, Optional, Any
from .utils import generate_sparse_input
from .functional import PlainLIFFunction, PlainLIFLocalZOMultiSample, PlainLIFLocalZO
from .samplers import BaseSampler, BaseOnceSampler


class LeakyPlain(nn.Module):

    def __init__(self, u_th, beta, batch_size):
        """
        Leaky integrate-and-fire neuron model with plain implementation, i.e. no considering sparsity
        Simply transform the input into dense mode then reshape the input into batch_size * num_neurons * num_steps * num_channels
        Then use the plain implementation of LIFFunction, and concatenate the output into batch_size * num_neurons * num_steps * num_channels

        Args:
            u_th: the threshold of membrane potential for firing
            beta: drop rate for membrane potential
            batch_size: batch size of input, should be the same as the batch size of the input data, so use drop_last=True in DataLoader
        """
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


class LeakeyZOPlain(nn.Module):

    def __init__(self, u_th, beta, batch_size, random_sampler: BaseSampler, sample_num, delta=0.01):
        """
        Leaky integrate-and-fire neuron model with plain implementation using ZO to approximate gradient

        Args:
            u_th: the threshold of membrane potential for firing
            beta: the drop rate for membrane potential
            batch_size: batch size of input, should be the same as the batch size of the input data, so use drop_last=True in DataLoader
            random_sampler: the random sampler used to generate random samples
            sample_num: the number of samples used to approximate gradient
            delta: the delta used in ZO
        """
        super(LeakeyZOPlain, self).__init__()
        self.batch_size = batch_size
        self.beta = beta
        self.u_th = u_th
        self.random_sampler: BaseSampler = random_sampler
        self.sample_num = sample_num
        self.delta = delta

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            random_tangents = self.random_sampler.generate_random_tangents(inputs.shape, self.batch_size, self.sample_num,
                                                                           device=inputs.device)
            outputs = PlainLIFLocalZOMultiSample.apply(inputs, self.batch_size, self.u_th,
                                                       self.beta, random_tangents, self.delta)
            return outputs

        # else
        inputs = inputs.dense()
        random_tangents = self.random_sampler.generate_random_tangents(inputs.shape, self.batch_size, self.sample_num,
                                                                       device=inputs.device)
        outputs = PlainLIFLocalZOMultiSample.apply(inputs, self.batch_size, self.u_th,
                                                   self.beta, random_tangents, self.delta)
        return spconv.SparseConvTensor.from_dense(outputs.transpose(1, 3))


class LeakyZOPlainOnce(nn.Module):
    """only sample once to accelerate the training process"""

    def __init__(self, u_th, beta, batch_size, random_sampler: BaseSampler, delta=0.01):
        """
        Leaky integrate-and-fire neuron model with plain implementation using ZO to approximate gradient

        Args:
            u_th: the threshold of membrane potential for firing
            beta: the drop rate for membrane potential
            batch_size: batch size of input, should be the same as the batch size of the input data, so use drop_last=True in DataLoader
            random_sampler: the random sampler used to generate random samples
            sample_num: the number of samples used to approximate gradient
            delta: the delta used in ZO
        """
        super(LeakyZOPlainOnce, self).__init__()
        self.batch_size = batch_size
        self.beta = beta
        self.u_th = u_th
        self.random_sampler: BaseOnceSampler = random_sampler
        self.delta = delta

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            random_tangents = self.random_sampler.generate_random_tangents(inputs.shape, self.batch_size,
                                                                           device=inputs.device)
            outputs = PlainLIFLocalZO.apply(inputs, self.batch_size, self.u_th, self.beta, random_tangents, self.delta)
            return outputs

        # else
        inputs = inputs.dense()
        random_tangents = self.random_sampler.generate_random_tangents(inputs.shape, self.batch_size,
                                                                       device=inputs.device)
        outputs = PlainLIFLocalZO.apply(inputs, self.batch_size, self.u_th, self.beta, random_tangents, self.delta)
        return spconv.SparseConvTensor.from_dense(outputs.transpose(1, 3))
