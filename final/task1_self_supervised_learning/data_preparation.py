import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from MLclf import MLclf  # 导入MLclf包来比较方便的拿到数据集

# 数据集存储路径
DATA_DIR = './data'

def download_and_preprocess_mini_imagenet(data_dir=DATA_DIR):
    """
    下载并预处理 mini-ImageNet 数据集。
    """
    # 从 MLclf 包获取 mini-ImageNet 数据集
    MLclf.miniimagenet_download(Download=True)

    # 将所有数据用于预训练
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset, val_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(
        ratio_train=0.8, ratio_val=0.1, seed_value=42, shuffle=True, transform=transform, save_clf_data=False
    )

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])

    return full_dataset

def download_and_preprocess_cifar100(data_dir=DATA_DIR):
    """
    下载并预处理 CIFAR-100 数据集。
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset

def get_dataloaders(batch_size, data_dir=DATA_DIR):
    """
    获取数据加载器。
    """
    mini_imagenet_dataset = download_and_preprocess_mini_imagenet(data_dir)
    train_cifar100, test_cifar100 = download_and_preprocess_cifar100(data_dir)

    mini_imagenet_loader = DataLoader(mini_imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader_cifar100 = DataLoader(train_cifar100, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_cifar100 = DataLoader(test_cifar100, batch_size=batch_size, shuffle=False, num_workers=4)

    return mini_imagenet_loader, train_loader_cifar100, test_loader_cifar100






if __name__ == "__main__":
    batch_size = 128
    mini_imagenet_loader, train_loader_cifar100, test_loader_cifar100 = get_dataloaders(batch_size)
    print("Data loaders ready!")
