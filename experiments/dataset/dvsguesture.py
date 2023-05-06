import tonic
from tonic import transforms
import torch
from torch.utils.data import DataLoader


def get_dataset(root, batch_size):
    sensor_size = tonic.datasets.DVSGesture.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=30000),
    ])

    print('start loading dataset')
    transform = torch.from_numpy
    dvsguesture_frame = tonic.datasets.DVSGesture(save_to=root, transform=frame_transform, train=False)
    print(dvsguesture_frame[0][0].shape)
    print('load into cache')
    dvsguesture_cached = tonic.DiskCachedDataset(dvsguesture_frame, cache_path='./cache/dvs', transform=transform)
    train_loader = DataLoader(dataset=dvsguesture_cached,
                              batch_size=batch_size, num_workers=4,
                              collate_fn=tonic.collation.PadTensors(batch_first=False))
    return train_loader


if __name__ == '__main__':
    train_loader = get_dataset('/home/zxh/data', 64)
    print(next(iter(train_loader))[0].shape)


