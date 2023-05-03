from torch import nn
import spconv.pytorch as spconv
from LocalZO.conv_models.neurons import LeakyPlain, LeakeyZOPlain, LeakyZOPlainOnce
from LocalZO.conv_models.samplers import BaseSampler, NormalSampler, NormalOnceSampler, BaseOnceSampler
import torch


class SimpleConvNet(nn.Module):

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


class SpconvNet(nn.Module):

    def __init__(self, beta=0.5, u_th=0.1, batch_size=64, algo=spconv.ConvAlgo.Native):
        """
        plain implementation of spiking convolutional network to compare the snntorch

        Args:
            beta: drop rate for membrane potential
            batch_size: batch size of input, should be the same as the batch size of the input data, so use drop_last=True in DataLoader
            u_th: the threshold of membrane potential for firing
            algo: algorithm for sparse convolution, default is Native
        """
        super(SpconvNet, self).__init__()
        self.conv1 = spconv.SparseConv2d(in_channels=2, out_channels=12, kernel_size=5, algo=algo, bias=True)
        self.pool1 = spconv.SparseMaxPool2d(kernel_size=2, stride=2, algo=algo, )
        self.ac1 = LeakyPlain(u_th, beta, batch_size)
        self.conv2 = spconv.SparseConv2d(in_channels=12, out_channels=32, kernel_size=5, algo=algo, bias=True)
        self.pool2 = spconv.SparseMaxPool2d(kernel_size=2, stride=2, algo=algo)
        self.ac2 = LeakyPlain(u_th, beta, batch_size)
        self.to_dense = spconv.ToDense()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(32 * 5 * 5, 10)
        self.ac3 = LeakyPlain(u_th, beta, batch_size)
        self.batch_size = batch_size

    def forward(self, inputs):
        assert isinstance(inputs, spconv.SparseConvTensor), 'inputs should be SparseConvTensor'
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.ac2(x)
        x = self.to_dense(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.ac3(x)
        x = x.view(-1, self.batch_size, 10)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path):
        net = SpconvNet()
        net.load_state_dict(torch.load(path))
        return net


class ExampleNetLocalZO(nn.Module):
    def __init__(self, batch_size, u_th, beta, conv_algorithm=spconv.ConvAlgo.MaskSplitImplicitGemm,
                 random_sampler: BaseSampler = NormalSampler, sample_num: int =5):
        super(ExampleNetLocalZO, self).__init__()
        self.sample_num = sample_num
        self.batch_size = batch_size
        self.conv_block1 = nn.Sequential(
            spconv.SparseConv2d(2, 16, 5, bias=True, algo=conv_algorithm),
            spconv.SparseBatchNorm(16, eps=1e-5, momentum=0.1),
            spconv.SparseMaxPool2d(2, stride=2),
            LeakeyZOPlain(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=random_sampler(), sample_num=sample_num)
        )
        self.conv_block2 = nn.Sequential(
            spconv.SparseConv2d(16, 32, 5, bias=True, algo=conv_algorithm),
            spconv.SparseBatchNorm(32, eps=1e-5, momentum=0.1),
            spconv.SparseMaxPool2d(2, stride=2),
            LeakeyZOPlain(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=random_sampler(), sample_num=sample_num)
        )
        self.to_dense = spconv.ToDense()  # convert the sparse tensor to dense tensor
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 5, 10),
            LeakyPlain(u_th=u_th, beta=beta, batch_size=batch_size)
        )

    def forward(self, x):
        assert isinstance(x, spconv.SparseConvTensor)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.to_dense(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(-1, self.batch_size, 10)  # reshape back into time, batch, *inputs
        return x


class ExampleNetLocalZOOnce(nn.Module):
    def __init__(self, batch_size, u_th, beta, conv_algorithm=spconv.ConvAlgo.Native,
                 random_sampler: BaseOnceSampler = NormalOnceSampler):
        """ similar to the prior one, but only sample once for each layer
        """
        super(ExampleNetLocalZOOnce, self).__init__()
        self.batch_size = batch_size
        self.conv_block1 = nn.Sequential(
            spconv.SparseConv2d(2, 16, 5, bias=True, algo=conv_algorithm),
            spconv.SparseBatchNorm(16, eps=1e-5, momentum=0.1),
            spconv.SparseMaxPool2d(2, stride=2),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=random_sampler(),)
        )
        self.conv_block2 = nn.Sequential(
            spconv.SparseConv2d(16, 32, 5, bias=True, algo=conv_algorithm),
            spconv.SparseBatchNorm(32, eps=1e-5, momentum=0.1),
            spconv.SparseMaxPool2d(2, stride=2),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=random_sampler(),)
        )
        self.to_dense = spconv.ToDense()  # convert the sparse tensor to dense tensor
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 5, 10),
            LeakyPlain(u_th=u_th, beta=beta, batch_size=batch_size)
        )

    def forward(self, x):
        assert isinstance(x, spconv.SparseConvTensor)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.to_dense(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(-1, self.batch_size, 10)  # reshape back into time, batch, *inputs
        return x
