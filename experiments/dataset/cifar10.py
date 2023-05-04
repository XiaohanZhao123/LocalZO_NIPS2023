import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from .utils import ConstantEncodingCollate
from torch.utils.data import DataLoader


def get_dataset(root, batch_size, encoding: str = 'constant', num_steps=6):
    assert encoding == 'constant', 'currently only constant encoding is supported'
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(224),
        transforms.ToTensor(),
        ])
    dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=ConstantEncodingCollate(num_steps=num_steps))
    return train_loader


if __name__ == '__main__':
    train_loader = get_dataset(root='/home/zxh/remote/async_fgd/experiment_resources/dataset/cifar10', batch_size=64,
                               encoding='constant', num_steps=6)
    print(next(iter(train_loader))[0].shape)


