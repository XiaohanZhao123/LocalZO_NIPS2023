import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import tonic
from tonic import transforms
import torch
from torch.utils.data import DataLoader
from experiments.models.simplenet import get_model
from snntorch.functional import accuracy_rate
from snntorch.utils import reset


def get_dataset(root, batch_size, time_window: int):
    sensor_size = tonic.datasets.DVSGesture.sensor_size

    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
    ])

    print('start loading dataset')
    dvsguesture_frame = tonic.datasets.DVSGesture(save_to=root, transform=frame_transform, train=False)
    print(dvsguesture_frame[0][0].shape)
    print('load into cache')
    train_loader = DataLoader(dataset=dvsguesture_frame,
                              batch_size=batch_size, num_workers=4,
                              collate_fn=tonic.collation.PadTensors(batch_first=False))
    return train_loader


if __name__ == '__main__':
    batch_size = 12
    train_loader = get_dataset('/home/zxh/data', batch_size)
    spconv_model, snntorch_model = get_model(batch_size, 1.0, 0.5, 11)
    print('inputs shape', next(iter(train_loader))[0].shape)
    print('start testing')
    spconv_model = spconv_model.cuda()
    snntorch_model = snntorch_model.cuda()
    spconv_model.eval()
    snntorch_model.eval()
    with torch.no_grad():
        reset(snntorch_model)
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.cuda()
            target = target.cuda()
            logits = spconv_model(inputs)
            acc = accuracy_rate(logits, target)
            print('spconv acc', acc)



