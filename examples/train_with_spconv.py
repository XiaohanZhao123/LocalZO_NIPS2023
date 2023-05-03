import time
from pathlib import Path
import sys
import os

gpu_id = '3'
sys.path.append(os.pardir)
print(sys.path)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
from LocalZO.conv_models.modules import LeakyPlain, DummyConvSNN
import torch
from torch import nn
from spconv import pytorch as spconv
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
import torchvision
import snntorch.functional as SF
from torch.nn.utils import clip_grad_norm_

criterion = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
criterion = SF.ce_rate_loss


class SpconvNet(nn.Module):

    def __init__(self, beta=0.5, u_th=0.1, batch_size=64):
        """
        plain implementation of spiking convolutional network

        Args:
            beta: drop rate for membrane potential
            batch_size: batch size of input, should be the same as the batch size of the input data, so use drop_last=True in DataLoader
            u_th: the threshold of membrane potential for firing
        """
        super(SpconvNet, self).__init__()
        algo = spconv.ConvAlgo.Native
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


def train(net,
          train_loader,
          test_loader,
          epochs=5,
          lr=1e-2,
          log_dir='./outputs/spconv',
          device='cuda', ):
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(epochs):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        for i, (inputs, labels) in enumerate(train_loader):
            # reshape to (time * batch_size, 28, 28, 2)
            start = time.time()
            inputs = inputs.view(-1, *inputs.shape[2:]).transpose(1, 3).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = spconv.SparseConvTensor.from_dense(inputs)
            outputs = net(inputs)
            loss_val = criterion(outputs, labels)
            clip_grad_norm_(net.parameters(), max_norm=5)
            loss_val.backward()
            optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            acc = (torch.argmax(torch.sum(outputs, dim=0), dim=1) == labels).sum().item() / labels.shape[0]
            if i % 10 == 0:
                print(f'epoch: {epoch}, iter: {i}, loss: {loss_val.item()}, time: {end - start}, acc: {acc}')
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = spconv.SparseConvTensor.from_dense(inputs)
                outputs = net(inputs)
                prediction = torch.argmax(torch.sum(outputs, dim=0), dim=1)
                correct += (prediction == labels).sum().item()

        print(f'epoch: {epoch}, test accuracy: {correct / total}')
        writer.add_scalar('train_loss', loss_val.item(), epoch)
        writer.add_scalar('test_accuracy', correct / total, epoch)


if __name__ == '__main__':
    batch_size = 64
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])

    transform = tonic.transforms.Compose([
        torch.from_numpy,
        torchvision.transforms.RandomRotation([-10, 10]),
    ])

    # load mnist dataset
    train_dataset = tonic.datasets.NMNIST(save_to='/home/zxh/data', train=True, transform=frame_transform)
    test_dataset = tonic.datasets.NMNIST(save_to='/home/zxh/data', train=False, transform=frame_transform)
    train_cached = DiskCachedDataset(train_dataset, cache_path='/home/zxh/data/train_cache', transform=transform)
    test_cached = DiskCachedDataset(test_dataset, cache_path='/home/zxh/data/test_cache', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, collate_fn=tonic.collation.PadTensors(batch_first=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             drop_last=True, collate_fn=tonic.collation.PadTensors(batch_first=False))

    # create network
    net = SpconvNet(u_th=0.8, beta=0.6, batch_size=batch_size)
    net = net.cuda()

    # training
    train(net, train_loader, test_loader, epochs=5, lr=2e-3, log_dir='./outputs/spconv', device='cuda')
    net.save('./outputs/spconv.pt')
