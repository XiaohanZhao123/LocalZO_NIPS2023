import snntorch
import torch
import torch.nn as nn
from snntorch import surrogate
from spconv import pytorch as spconv

from LocalZO.conv_models.neurons import LeakyPlain, LeakyZOPlainOnce
from LocalZO.conv_models.samplers import NormalOnceSampler

spike_grad = surrogate.sigmoid()


class VGGSNNTorch(nn.Module):

    def __init__(self, beta, u_th, in_channel, num_class, ):
        """
        implement the vgg network with snntorch
        
        Args:
            beta: the drop rate of membrane potential
            u_th: the threshold of membrane potential for firing
            in_channel: the number of input channel
            num_class: the number of output class
        """
        super(VGGSNNTorch, self).__init__()
        self.in_channel = in_channel
        self.num_class = num_class

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=64, kernel_size=3, padding=1),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 3 * 3, out_features=self.num_class),
            snntorch.Leaky(beta=beta, threshold=u_th, spike_grad=spike_grad, init_hidden=True, output=True),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGSpconv(nn.Module):

    def __init__(self, in_channel, num_class, batch_size, u_th, beta, delta=0.01, sampler=NormalOnceSampler,
                 algo=spconv.ConvAlgo.Native, **kwarg):
        super(VGGSpconv, self).__init__()
        self.in_channel = in_channel
        self.num_class = num_class
        self.batch_size = batch_size

        self.conv_layers = nn.Sequential(
            spconv.SparseConv2d(in_channel, 64, 3, padding=1, algo=algo),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=sampler(), delta=delta),
            spconv.SparseConv2d(64, 128, 3, padding=1, algo=algo),
            spconv.SparseMaxPool2d(2, 2, algo=algo),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=sampler(), delta=delta),

            spconv.SparseConv2d(128, 256, 3, padding=1, algo=algo),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=sampler(), delta=delta),
            spconv.SparseConv2d(256, 512, 3, padding=1, algo=algo),
            spconv.SparseMaxPool2d(2, 2, algo=algo),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=sampler(), delta=delta),

            spconv.SparseConv2d(512, 512, 3, padding=1, algo=algo),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=sampler(), delta=delta),
            spconv.SparseConv2d(512, 512, 3, padding=1, algo=algo),
            spconv.SparseMaxPool2d(2, 2, algo=algo),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=sampler(), delta=delta),

            spconv.SparseConv2d(512, 512, 3, padding=1, algo=algo),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=sampler(), delta=delta),
            spconv.SparseConv2d(512, 512, 3, padding=1, algo=algo),
            spconv.SparseMaxPool2d(2, 2, algo=algo),
            LeakyZOPlainOnce(u_th=u_th, beta=beta, batch_size=batch_size, random_sampler=sampler(), delta=delta),

        )

        self.classifier = nn.Sequential(
            spconv.ToDense(),
            nn.Flatten(),
            nn.Linear(in_features=512 * 3 * 3, out_features=self.num_class),
            LeakyPlain(u_th=u_th, beta=beta, batch_size=batch_size),
        )

    def forward(self, x):
        # reshape the x from [time, batch, *pic_shape] into [time*batch, *pic_shape]
        x = x.view(-1, *x.shape[2:]).transpose(1, 3)  # move channel to the last one
        x = spconv.SparseConvTensor.from_dense(x)
        x = self.conv_layers(x)
        x = self.classifier(x)
        x = x.view(-1, self.batch_size, self.num_class)
        return x


def get_models(batch_size, beta, num_class, u_th, in_channel=2):
    torchnet = VGGSNNTorch(beta=beta, in_channel=in_channel, u_th=u_th, num_class=num_class).cuda()
    spconvnet = VGGSpconv(in_channel=in_channel, u_th=u_th, num_class=num_class, beta=beta,
                          batch_size=batch_size).cuda()
    return spconvnet, torchnet


if __name__ == '__main__':
    spconv_net, torchnet = get_models(batch_size=1, beta=0.3, num_class=10, u_th=1.0)
    inputs = torch.randn(1, 2, 128, 128).cuda()
    torchnet(inputs)
    inputs = torch.randn(3, 1, 2, 128, 128).cuda()
    outputs = spconv_net(inputs)
    print(outputs.shape)
