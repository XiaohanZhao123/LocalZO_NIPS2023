from torch import nn
import spconv.pytorch as spconv
from LocalZO.conv_models.neurons import LeakyPlain
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
        self.pool1 = spconv.SparseMaxPool2d(kernel_size=2, stride=2, algo=algo,)
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