import tonic
from tonic import transforms
import torch
from torch.utils.data import DataLoader


def get_dataset(root, batch_size):
    sensor_size = tonic.datasets.DVSGesture.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
    ])

    transform = torch.from_numpy
    dvsguesture_frame = tonic.datasets.DVSGesture(save_to=root, transform=frame_transform, train=True)
    dvsguesture_cached = tonic.DiskCachedDataset(dvsguesture_frame, cache_path='./cache/dvs_guesture', transform=transform)
    train_loader = DataLoader(dataset=dvsguesture_cached, batch_size=batch_size, num_workers=4, pin_memory=True)
    return train_loader




