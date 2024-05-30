import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data_preparation import get_dataloaders
from model import get_resnet18_model
from pretraining import MoCo, train_moco
from training_finetuning import train_cifar100, validate_cifar100, finetune

# 设置参数
batch_size = 128
num_epochs_pretrain = 200
num_epochs_finetune = 100
learning_rate_list = [0.1, 0.01, 0.001]

# 获取数据加载器
mini_imagenet_loader, train_loader_cifar100, test_loader_cifar100 = get_dataloaders(batch_size)

# MoCo 预训练
print("Starting MoCo pre-training on mini-ImageNet...")
moco_model = MoCo(base_encoder=get_resnet18_model, dim=128, K=65536, m=0.999, T=0.07, mlp=False).cuda()
moco_optimizer = optim.SGD(moco_model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
moco_criterion = nn.CrossEntropyLoss().cuda()
moco_writer = SummaryWriter(log_dir='logs/moco')

for epoch in range(num_epochs_pretrain):
    train_loss = train_moco(mini_imagenet_loader, moco_model, moco_criterion, moco_optimizer, epoch, moco_writer)
    print(f'Epoch [{epoch+1}/{num_epochs_pretrain}], MoCo Loss: {train_loss:.4f}')
moco_writer.close()

# 获取 MoCo 预训练的 encoder_q
moco_pretrained_model = moco_model.encoder_q

# 微调和评估 MoCo 预训练模型
print("Finetuning MoCo pre-trained model on CIFAR-100...")
best_lr = finetune(train_loader_cifar100, test_loader_cifar100, moco_pretrained_model, learning_rate_list, num_epochs_finetune)
print(f'Best learning rate for fine-tuning: {best_lr}')

# 使用最佳学习率训练和评估 MoCo 预训练模型
print("Training MoCo pre-trained model on CIFAR-100 with best learning rate...")
finetune_optimizer = optim.Adam([
    {'params': moco_pretrained_model.parameters(), 'lr': best_lr / 10},
    {'params': moco_pretrained_model.fc.parameters(), 'lr': best_lr}
])
finetune_writer = SummaryWriter(log_dir='logs/moco_finetune')
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs_finetune):
    train_loss, train_acc = train_cifar100(train_loader_cifar100, moco_pretrained_model, criterion, finetune_optimizer, epoch, finetune_writer)
    val_loss, val_acc = validate_cifar100(test_loader_cifar100, moco_pretrained_model, criterion, epoch, finetune_writer)
    print(f'Epoch [{epoch+1}/{num_epochs_finetune}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
finetune_writer.close()

# 加载 ImageNet 预训练模型
print("Evaluating ImageNet pre-trained model on CIFAR-100...")
imagenet_pretrained_model = get_resnet18_model(pretrained=True, num_classes=100).cuda()
imagenet_best_lr = finetune(train_loader_cifar100, test_loader_cifar100, imagenet_pretrained_model, learning_rate_list, num_epochs_finetune)

# 使用最佳学习率训练和评估 ImageNet 预训练模型
finetune_optimizer = optim.Adam([
    {'params': imagenet_pretrained_model.parameters(), 'lr': imagenet_best_lr / 10},
    {'params': imagenet_pretrained_model.fc.parameters(), 'lr': imagenet_best_lr}
])
imagenet_finetune_writer = SummaryWriter(log_dir='logs/imagenet_finetune')

for epoch in range(num_epochs_finetune):
    train_loss, train_acc = train_cifar100(train_loader_cifar100, imagenet_pretrained_model, criterion, finetune_optimizer, epoch, imagenet_finetune_writer)
    val_loss, val_acc = validate_cifar100(test_loader_cifar100, imagenet_pretrained_model, criterion, epoch, imagenet_finetune_writer)
    print(f'Epoch [{epoch+1}/{num_epochs_finetune}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
imagenet_finetune_writer.close()

# 训练和评估随机初始化模型
print("Training randomly initialized model on CIFAR-100...")
random_init_model = get_resnet18_model(pretrained=False, num_classes=100).cuda()
random_best_lr = finetune(train_loader_cifar100, test_loader_cifar100, random_init_model, learning_rate_list, num_epochs_finetune)

# 使用最佳学习率训练和评估随机初始化模型
finetune_optimizer = optim.Adam([
    {'params': random_init_model.parameters(), 'lr': random_best_lr / 10},
    {'params': random_init_model.fc.parameters(), 'lr': random_best_lr}
])
random_finetune_writer = SummaryWriter(log_dir='logs/random_finetune')

for epoch in range(num_epochs_finetune):
    train_loss, train_acc = train_cifar100(train_loader_cifar100, random_init_model, criterion, finetune_optimizer, epoch, random_finetune_writer)
    val_loss, val_acc = validate_cifar100(test_loader_cifar100, random_init_model, criterion, epoch, random_finetune_writer)
    print(f'Epoch [{epoch+1}/{num_epochs_finetune}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
random_finetune_writer.close()

print("Experiment completed!")
