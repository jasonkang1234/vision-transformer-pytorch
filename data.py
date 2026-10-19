import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    image_size: int = 32,
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    train_loader: DataLoader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)  # type: ignore[type-arg]
    test_loader: DataLoader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)  # type: ignore[type-arg]
    return train_loader, test_loader
