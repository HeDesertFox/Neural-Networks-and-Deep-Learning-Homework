import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from MLclf import MLclf

# 数据集存储路径
DATA_DIR = './data'

def download_and_preprocess_imagenet(data_type='mini', data_dir=DATA_DIR):
    """
    下载并预处理 ImageNet 数据集（mini 或 tiny）。

    参数:
    data_type (str): 数据集类型 ('mini' 或 'tiny')。
    data_dir (str): 数据集存储路径。

    返回:
    train_dataset, val_dataset, test_dataset (torch.utils.data.TensorDataset): 预处理后的 ImageNet 数据集。
    """
    if data_type == 'mini':
        imagenet_dir = os.path.join(data_dir, 'miniimagenet/')
        MLclf.download_dir = data_dir
        if not os.path.exists(imagenet_dir):
            MLclf.miniimagenet_download(Download=True)
    elif data_type == 'tiny':
        imagenet_dir = os.path.join(data_dir, 'tiny-imagenet-200/')
        MLclf.download_dir_tinyIN = data_dir
        if not os.path.exists(imagenet_dir):
            MLclf.tinyimagenet_download(Download=True)
    else:
        raise ValueError("data_type 参数必须是 'mini' 或 'tiny'")

    # 定义数据预处理的转换操作
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 转换数据集格式并返回
    if data_type == 'mini':
        train_dataset, val_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(
            data_dir=imagenet_dir,
            ratio_train=0.99,
            ratio_val=0,
            seed_value=None,
            shuffle=True,
            transform=transform,
            save_clf_data=True
        )
    elif data_type == 'tiny':
        train_dataset, val_dataset, test_dataset = MLclf.tinyimagenet_clf_dataset(
            data_dir=imagenet_dir,
            ratio_train=0.99,
            ratio_val=0,
            seed_value=None,
            shuffle=True,
            transform=transform,
            save_clf_data=True
        )

    return train_dataset, val_dataset, test_dataset

def download_and_preprocess_cifar100(data_dir=DATA_DIR):
    """
    下载并预处理 CIFAR-100 数据集。

    参数:
    data_dir (str): 数据集存储路径。

    返回:
    train_dataset (torchvision.datasets.CIFAR100): CIFAR-100 训练集。
    test_dataset (torchvision.datasets.CIFAR100): CIFAR-100 测试集。
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

def get_dataloaders(batch_size, data_type='mini', data_dir=DATA_DIR):
    """
    获取数据加载器。

    参数:
    batch_size (int): 每个批次的数据量。
    data_type (str): 数据集类型 ('mini' 或 'tiny')。
    data_dir (str): 数据集存储路径。

    返回:
    imagenet_loader (torch.utils.data.DataLoader): ImageNet 数据加载器。
    train_loader_cifar100 (torch.utils.data.DataLoader): CIFAR-100 训练集数据加载器。
    test_loader_cifar100 (torch.utils.data.DataLoader): CIFAR-100 测试集数据加载器。
    """
    train_imagenet, val_imagenet, test_imagenet = download_and_preprocess_imagenet(data_type=data_type, data_dir=data_dir)
    train_cifar100, test_cifar100 = download_and_preprocess_cifar100(data_dir)

    imagenet_loader = DataLoader(torch.utils.data.ConcatDataset([train_imagenet, val_imagenet, test_imagenet]), batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader_cifar100 = DataLoader(train_cifar100, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_cifar100 = DataLoader(test_cifar100, batch_size=batch_size, shuffle=False, num_workers=4)

    return imagenet_loader, train_loader_cifar100, test_loader_cifar100
