import os
import tarfile
import urllib.request
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 下载Mini-ImageNet数据集的函数
def download_mini_imagenet(data_dir):
    url = "http://example.com/mini-imagenet.tar.gz"  # 替换为实际的Mini-ImageNet下载链接
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    tar_path = os.path.join(data_dir, "mini-imagenet.tar.gz")
    if not os.path.exists(tar_path):
        print("Downloading Mini-ImageNet dataset...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")

    # 解压缩
    if not os.path.exists(os.path.join(data_dir, "mini-imagenet")):
        print("Extracting Mini-ImageNet dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")

# Mini-ImageNet自定义数据集
class MiniImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label in os.listdir(self.root):
            label_path = os.path.join(self.root, label)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    if img_file.endswith(".jpg"):
                        self.data.append(os.path.join(label_path, img_file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# CIFAR-100数据加载和预处理
def get_cifar100_data(data_dir, batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# 获取Mini-ImageNet数据加载器
def get_mini_imagenet_data(data_dir, batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = MiniImageNetDataset(root=os.path.join(data_dir, "mini-imagenet"), transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader

if __name__ == "__main__":
    data_dir = "./data"

    # 下载并预处理Mini-ImageNet数据集
    download_mini_imagenet(data_dir)
    mini_imagenet_loader = get_mini_imagenet_data(data_dir)

    # 下载并预处理CIFAR-100数据集
    cifar100_train_loader, cifar100_test_loader = get_cifar100_data(data_dir)

    print("Data preparation complete.")
