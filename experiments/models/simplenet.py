import torch
import spconv.pytorch as spconv
from torch import nn
from LocalZO.conv_models.neurons import LeakyPlain, LeakyZOPlainOnce
import snntorch
from snntorch import surrogate

surrogate_grad = surrogate.sigmoid()


class SimpleNetSpconv(nn.Module):
    def __init__(self, u_th, beta, batch_size, num_class, algo=spconv.ConvAlgo.Native):
        super().__init__()
        self.batch_size = batch_size
        self.num_class = num_class

        self.pool1 = spconv.SparseMaxPool2d(kernel_size=4, stride=4, algo=algo)
        self.conv1 = spconv.SparseConv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1, algo=algo)
        self.snn1 = LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size)
        self.conv2 = spconv.SparseConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, algo=algo)
        self.snn2 = LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size)
        self.pool2 = spconv.SparseAvgPool2d(kernel_size=2, stride=2, algo=algo)
        self.snn3 = LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size)
        self.conv3 = spconv.SparseConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, algo=algo)
        self.pool3 = spconv.SparseAvgPool2d(kernel_size=2, stride=2, algo=algo)
        self.to_dense = spconv.ToDense()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8192, 11)
        self.snn_out = LeakyPlain(u_th=u_th, beta=beta, batch_size=batch_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=0, end_dim=1).transpose(1, 3)  # reshape into batch_size * timestep
        x = spconv.SparseConvTensor.from_dense(x)
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.snn1(x)
        x = self.conv2(x)
        x = self.snn2(x)
        x = self.pool2(x)
        x = self.snn3(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.to_dense(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.snn_out(x)
        x = x.view(-1, self.batch_size, self.num_class)  # reshape into batch_size , timestep , num_class
        return x


class SimpleNetSNNtorch(nn.Module):
    def __init__(self, u_th, beta):
        super().__init__()
        self.poo1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.snn1 = snntorch.Leaky(threshold=u_th, beta=beta, spike_grad=surrogate_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.snn2 = snntorch.Leaky(threshold=u_th, beta=beta, spike_grad=surrogate_grad, init_hidden=True)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.snn3 = snntorch.Leaky(threshold=u_th, beta=beta, spike_grad=surrogate_grad, init_hidden=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8192, 11)
        self.snn_out = snntorch.Leaky(threshold=u_th, beta=beta, spike_grad=surrogate_grad, init_hidden=True,
                                      output=True)

    def forward(self, x):
        x = self.poo1(x)
        x = self.conv1(x)
        x = self.snn1(x)
        x = self.conv2(x)
        x = self.snn2(x)
        x = self.pool2(x)
        x = self.snn3(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.snn_out(x)
        return x


def get_model(batch_size, u_th, beta, num_class):
    simplenet_spconv = SimpleNetSpconv(u_th=u_th, beta=beta, batch_size=batch_size, num_class=num_class).cuda()
    simplenet_snntorch = SimpleNetSNNtorch(u_th=u_th, beta=beta).cuda()
    return simplenet_spconv, simplenet_snntorch


if __name__ == '__main__':
    xs = torch.randn(3, 4, 2, 128, 128).cuda()
    spconv_model, snntorch_model = get_model(u_th=0.5, beta=0.2, batch_size=4, num_class=11)
    spconv_model = spconv_model.cuda()
    snntorch_model = snntorch_model.cuda()

    for x in xs:
        spk, mem = snntorch_model(x)
        print(spk.shape)

    out = spconv_model(xs)
    print(out.shape)
