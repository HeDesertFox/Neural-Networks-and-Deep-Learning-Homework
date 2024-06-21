import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import urllib.request
import zipfile
import tarfile


def download_and_extract_tiny_imagenet(data_dir='./data/tiny-imagenet'):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(file_path):
        print('Downloading Tiny ImageNet...')
        urllib.request.urlretrieve(url, file_path)
        print('Download complete.')
    else:
        print('Tiny ImageNet zip file already exists.')

    extract_path = os.path.join(data_dir, 'tiny-imagenet-200')
    if not os.path.exists(extract_path):
        print('Extracting Tiny ImageNet...')
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print('Extraction complete.')
    else:
        print('Tiny ImageNet directory already exists.')

    print('Tiny ImageNet is ready to use.')
    return extract_path

def download_and_extract_caltech256(data_dir='./data/caltech256'):
    url = 'https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1'
    filename = '256_ObjectCategories.tar'
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(file_path):
        print('Downloading Caltech-256...')
        urllib.request.urlretrieve(url, file_path)
        print('Download complete.')
    else:
        print('Caltech-256 tar file already exists.')

    extract_path = os.path.join(data_dir, '256_ObjectCategories')
    if not os.path.exists(extract_path):
        print('Extracting Caltech-256...')
        with tarfile.open(file_path, 'r') as tar_ref:
            tar_ref.extractall(data_dir)
        print('Extraction complete.')
    else:
        print('Caltech-256 directory already exists.')

    print('Caltech-256 is ready to use.')
    return extract_path


def download_and_preprocess_cifar100(data_dir='./data'):
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
