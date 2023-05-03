gpu_idx = '3'
import os
import tonic
from tonic import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
from torch import nn
from spconv import pytorch as spconv  # pay attention to the spconv.pytorch
from conv_models.neurons import LeakyPlain
from snntorch import functional as SF
import torch
import torchvision
from torch.utils.data import DataLoader
from conv_models.neurons import LeakeyZOPlain
from conv_models.samplers import NormalSampler, BaseSampler
from LocalZO.conv_models.models.examples import ExampleNetLocalZO, ExampleNetLocalZOOnce
from torch.utils.tensorboard import SummaryWriter
import time
from snntorch import functional as SF
import torch

# hyper parameters
batch_size = 128
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
acc_fn = SF.accuracy_rate


def get_nminist():
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                          transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
                                          ])
    trainset = tonic.datasets.NMNIST(save_to='/home/zxh/data', train=True, transform=frame_transform)
    testset = tonic.datasets.NMNIST(save_to='/home/zxh/data', train=False, transform=frame_transform)
    print('trainset size:', len(trainset), 'testset size:', len(testset), 'type of dataset', type(trainset[0][0]),
          'shape', trainset[0][0].shape)

    cache_transform = tonic.transforms.Compose([torch.from_numpy, torchvision.transforms.RandomRotation([-10, 10]), ])

    epoch_num = 5
    cached_trainset = tonic.DiskCachedDataset(trainset, cache_path='./data/cache', transform=cache_transform)
    cached_testset = tonic.DiskCachedDataset(testset, cache_path='./data/cache', )

    train_loader = DataLoader(cached_trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                              collate_fn=tonic.collation.PadTensors(batch_first=False))
    test_loader = DataLoader(cached_testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True,
                             collate_fn=tonic.collation.PadTensors(batch_first=False))

    return train_loader, test_loader


def train(net, epoch_num, train_loader, test_loader):
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
    num_iter = 51
    time_list = []
    with SummaryWriter(comment='spconv', log_dir='./output/spconv') as writer:
        for epoch in range(epoch_num):
            for i, (inputs, labels) in enumerate(train_loader):
                start = time.time()
                inputs = inputs.view(-1, *inputs.shape[2:]).transpose(1, 3).cuda()
                inputs = spconv.SparseConvTensor.from_dense(inputs)
                labels = labels.cuda()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                end = time.time()

                if i > 0:
                    time_list.append(end - start)

                if i == num_iter:
                    print('average time', sum(time_list) / len(time_list))
                    break
                if i % 10 == 0:
                    writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + i)
                    writer.add_scalar('acc', acc_fn(outputs, labels).item(), epoch * len(train_loader) + i)
                    print('epoch', epoch, 'batch', i, 'loss', loss.item(), 'acc', acc_fn(outputs, labels).item(),
                          'time', end - start)

            acc = 0
            loss_val = 0
            for inputs, labels in test_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = net(inputs)
                loss_val += loss_fn(outputs, labels).item()
                acc += acc_fn(outputs, labels).item()

            writer.add_scalar('test loss', loss_val / len(test_loader), epoch)
            writer.add_scalar('test acc', acc / len(test_loader), epoch)
            print('epoch', epoch, 'test loss', loss_val / len(test_loader), 'test acc', acc / len(test_loader))


if __name__ == '__main__':
    net = ExampleNetLocalZOOnce(batch_size=batch_size, u_th=1.0, beta=0.5).cuda()
    print(net)
    for name, param in net.named_parameters():
        print(name, param.shape)
    train_loader, test_loader = get_nminist()
    train(net, 5, train_loader, test_loader)
