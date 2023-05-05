import tonic
from tonic import transforms
import torch
from torch.utils.data import DataLoader


def get_dataset(root, batch_size):
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])

    transform = torch.from_numpy
    dvscifar10_frame = tonic.datasets.CIFAR10DVS(save_to=root, transform=frame_transform)
    dvscifar10_cached = tonic.DiskCachedDataset(dvscifar10_frame, cache_path='./cache/dvs', transform=transform)
    train_loader = DataLoader(dataset=dvscifar10_cached, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader




