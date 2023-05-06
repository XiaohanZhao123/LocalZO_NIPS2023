import tonic
from tonic import transforms
import torch
from torch.utils.data import DataLoader
from snntorch.utils import data_subset


def get_dataset(root, batch_size):
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=3000),
        torch.from_numpy
    ])

    # transform = torch.from_numpy
    dvscifar10_frame = tonic.datasets.CIFAR10DVS(save_to=root, transform=frame_transform)
    print('the shape of input', dvscifar10_frame[0][0].shape)
    dvscifar10_cached = tonic.DiskCachedDataset(dvscifar10_frame, cache_path='./cache/dvscifar')
    train_loader = DataLoader(dataset=dvscifar10_cached,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True,
                              collate_fn=tonic.collation.PadTensors(batch_first=False))
    return train_loader


if __name__ == '__main__':
    train_loader = get_dataset('/home/zxh/data', 64)
    print(next(iter(train_loader))[0].shape)


