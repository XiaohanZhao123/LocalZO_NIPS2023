import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataset(root, batch_size):
    transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    cifar100_train = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=cifar100_train,
                              num_workers=4,
                              pin_memory=True,
                              batch_size=batch_size)
    return train_loader


if __name__ == '__main__':
    train_loader = get_dataset('/home/zxh/data/cifar100', 64)
    print('input shape', next(iter(train_loader))[0].shape)
