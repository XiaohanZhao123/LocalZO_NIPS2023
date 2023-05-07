import tonic
import torch
from tonic import transforms
from torch.utils.data import DataLoader


def get_dataset(root, batch_size):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])

    transform = torch.from_numpy
    nminist_frame = tonic.datasets.NMNIST(save_to=root, transform=frame_transform, train=True)
    nminist_cached = tonic.DiskCachedDataset(nminist_frame, cache_path='./cache/nminist', transform=transform)
    train_loader = DataLoader(dataset=nminist_cached, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader
