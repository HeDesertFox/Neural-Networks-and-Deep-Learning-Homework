import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
from data_preparation import download_and_preprocess_imagenet, download_and_preprocess_cifar100, save_dataset_to_folder
from model import get_resnet18_model
from training_finetuning import train, tune_hyperparameters

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集存储路径
DATA_DIR = './data'
OUTPUT_DIR = './output'

# 创建输出目录
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 下载并预处理ImageNet数据集
imagenet_dataset = download_and_preprocess_imagenet(data_type='mini', data_dir=DATA_DIR)

# 保存ImageNet数据集到文件系统
imagenet_train_dir = os.path.join(OUTPUT_DIR, 'train_imagenet')
save_dataset_to_folder(imagenet_dataset, imagenet_train_dir)

# 使用subprocess调用MoCo进行预训练
subprocess.run([
    'python', 'main_moco.py',
    imagenet_train_dir,
    '-a', 'resnet18',
    '--epochs', '200',
    '--batch-size', '256',
    '--lr', '0.03',
    '--mlp',
    '--moco-t', '0.2',
    '--aug-plus',
    '--cos'
])

# 获取CIFAR-100数据集
train_cifar100, test_cifar100 = download_and_preprocess_cifar100(data_dir=DATA_DIR)

# 创建DataLoader
batch_size = 64
train_loader_cifar100 = DataLoader(train_cifar100, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader_cifar100 = DataLoader(test_cifar100, batch_size=batch_size, shuffle=False, num_workers=4)

# 定义超参数范围
lr_values = [0.001, 0.01, 0.1]
weight_decay_values = [1e-4, 1e-3, 1e-2]
epochs = 100

# 实验1：使用MoCo预训练模型
moco_model = get_resnet18_model(pretrained=False, num_classes=100)
checkpoint = torch.load('checkpoint_0199.pth.tar')
moco_model.load_state_dict(checkpoint['state_dict'])

print("Tuning hyperparameters for MoCo pre-trained model")
best_hyperparams_moco = tune_hyperparameters(moco_model, train_loader_cifar100, test_loader_cifar100, epochs, lr_values, weight_decay_values, device)
print(f"Best hyperparameters for MoCo pre-trained model: {best_hyperparams_moco}")

# 使用最佳超参数进行最终训练
optimizer_moco = optim.Adam(moco_model.parameters(), lr=best_hyperparams_moco[0], weight_decay=best_hyperparams_moco[1])
criterion = nn.CrossEntropyLoss()
writer_moco = SummaryWriter()

print("Final training with best hyperparameters for MoCo pre-trained model")
train(moco_model.to(device), train_loader_cifar100, test_loader_cifar100, optimizer_moco, criterion, epochs, device, writer=writer_moco)

# 实验2：使用ImageNet预训练模型
imagenet_model = get_resnet18_model(pretrained=True, num_classes=100)

print("Tuning hyperparameters for ImageNet pre-trained model")
best_hyperparams_imagenet = tune_hyperparameters(imagenet_model, train_loader_cifar100, test_loader_cifar100, epochs, lr_values, weight_decay_values, device)
print(f"Best hyperparameters for ImageNet pre-trained model: {best_hyperparams_imagenet}")

# 使用最佳超参数进行最终训练
optimizer_imagenet = optim.Adam(imagenet_model.parameters(), lr=best_hyperparams_imagenet[0], weight_decay=best_hyperparams_imagenet[1])
writer_imagenet = SummaryWriter()

print("Final training with best hyperparameters for ImageNet pre-trained model")
train(imagenet_model.to(device), train_loader_cifar100, test_loader_cifar100, optimizer_imagenet, criterion, epochs, device, writer=writer_imagenet)

# 实验3：使用随机初始化模型
random_model = get_resnet18_model(pretrained=False, num_classes=100)

print("Tuning hyperparameters for randomly initialized model")
best_hyperparams_random = tune_hyperparameters(random_model, train_loader_cifar100, test_loader_cifar100, epochs, lr_values, weight_decay_values, device)
print(f"Best hyperparameters for randomly initialized model: {best_hyperparams_random}")

# 使用最佳超参数进行最终训练
optimizer_random = optim.Adam(random_model.parameters(), lr=best_hyperparams_random[0], weight_decay=best_hyperparams_random[1])
writer_random = SummaryWriter()

print("Final training with best hyperparameters for randomly initialized model")
train(random_model.to(device), train_loader_cifar100, test_loader_cifar100, optimizer_random, criterion, epochs, device, writer=writer_random)
