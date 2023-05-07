import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def get_dataset(root, batch_size, ):
    transform = transforms.Compose([
        transforms.CenterCrop(48),
        transforms.Resize(48),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True, )
    return train_loader
