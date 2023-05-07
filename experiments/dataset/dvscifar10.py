import tonic
import torch
from tonic import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Resize


def get_dataset(root, batch_size, time_window):
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
        torch.from_numpy,
        Resize(48)
    ])

    # transform = torch.from_numpy
    dvscifar10_frame = tonic.datasets.CIFAR10DVS(save_to=root, transform=frame_transform)
    print('the shape of input', dvscifar10_frame[0][0].shape)
    train_loader = DataLoader(dataset=dvscifar10_frame,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True,
                              collate_fn=tonic.collation.PadTensors(batch_first=False))
    return train_loader


if __name__ == '__main__':
    train_loader = get_dataset('/home/zxh/data', 64, time_window=6000)
    print(next(iter(train_loader))[0].shape)
