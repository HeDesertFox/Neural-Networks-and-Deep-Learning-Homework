import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data_preparation import get_dataloaders
from model import get_resnet18_model, add_fc_layer
from pretraining import MoCo, train_moco
from training_finetuning import train_and_validate, finetune

# 设置参数
batch_size = 128
num_epochs_pretrain = 200
num_epochs_finetune = 100
num_classes = 100
learning_rate_list = [0.1, 0.01, 0.001, 30]  # 注意一下不寻常的学习率

# 获取数据加载器
dataset_type = 'tiny'  # 可以选择 'tiny' 或 'mini'
imagenet_loader, train_loader_cifar100, test_loader_cifar100 = get_dataloaders(batch_size, data_type=dataset_type)

# # 保存数据集对象
# torch.save(imagenet_loader.dataset, 'imagenet_loader.pth')
# torch.save(train_loader_cifar100.dataset, 'train_loader_cifar100.pth')
# torch.save(test_loader_cifar100.dataset, 'test_loader_cifar100.pth')

# # 重新加载数据集对象的示例代码（注释掉，仅供参考）
# imagenet_dataset = torch.load('imagenet_loader.pth')
# train_dataset_cifar100 = torch.load('train_loader_cifar100.pth')
# test_dataset_cifar100 = torch.load('test_loader_cifar100.pth')
# imagenet_loader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# train_loader_cifar100 = DataLoader(train_dataset_cifar100, batch_size=batch_size, shuffle=True, num_workers=4)
# test_loader_cifar100 = DataLoader(test_dataset_cifar100, batch_size=batch_size, shuffle=False, num_workers=4)

# MoCo 预训练
print(f"Starting MoCo pre-training on {dataset_type}-ImageNet...")
moco_model = MoCo(base_encoder=get_resnet18_model, dim=128, K=65536, m=0.999, T=0.07, mlp=False).cuda()
moco_optimizer = optim.SGD(moco_model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
moco_criterion = nn.CrossEntropyLoss().cuda()
moco_writer = SummaryWriter(log_dir='logs/moco')

for epoch in range(num_epochs_pretrain):
    train_loss = train_moco(imagenet_loader, moco_model, moco_criterion, moco_optimizer, epoch, moco_writer)
    print(f'Epoch [{epoch+1}/{num_epochs_pretrain}], MoCo Loss: {train_loss:.4f}')
moco_writer.close()

# 保存 MoCo 预训练模型权重
torch.save(moco_model.state_dict(), 'moco_pretrained.pth')

# 重新导入模型权重的语句
# moco_model.load_state_dict(torch.load('moco_pretrained.pth'))

# 获取 MoCo 预训练的 encoder_q
moco_pretrained_model = moco_model.encoder_q
full_moco_pretrained_model = add_fc_layer(moco_pretrained_model, num_classes)

# 微调和评估 MoCo 预训练模型
print("Finetuning MoCo pre-trained model on CIFAR-100...")
best_lr = finetune(train_loader_cifar100, test_loader_cifar100, full_moco_pretrained_model, learning_rate_list, num_epochs_finetune)
print(f'Best learning rate for fine-tuning: {best_lr}')

#可以在这里直接设置学习率
best_lr = 2e-3

# 使用最佳学习率训练和评估 MoCo 预训练模型
print("Training MoCo pre-trained model on CIFAR-100 with best learning rate...")
finetune_optimizer = optim.Adam([
    {'params': full_moco_pretrained_model.parameters(), 'lr': best_lr / 10},
    {'params': full_moco_pretrained_model[-1].parameters(), 'lr': best_lr}
])
finetune_writer = SummaryWriter(log_dir='logs/moco_finetune')
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs_finetune):
    train_loss, train_acc, val_loss, val_acc = train_and_validate(train_loader_cifar100, test_loader_cifar100, full_moco_pretrained_model, criterion, finetune_optimizer, epoch, finetune_writer)
    print(f'Epoch [{epoch+1}/{num_epochs_finetune}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
finetune_writer.close()

# 保存 MoCo 预训练并微调后的模型权重
torch.save(full_moco_pretrained_model.state_dict(), 'moco_finetuned.pth')

# 重新导入模型权重的语句
# full_moco_pretrained_model.load_state_dict(torch.load('moco_finetuned.pth'))

# 评估 ImageNet 预训练模型
print("Evaluating ImageNet pre-trained model on CIFAR-100...")
imagenet_pretrained_model = get_resnet18_model(pretrained=True).cuda()
full_imagenet_pretrained_model = add_fc_layer(imagenet_pretrained_model, num_classes)

#可以在这里直接设置学习率
imagenet_best_lr = 2e-3

# 使用最佳学习率训练和评估 ImageNet 预训练模型
print("Training ImageNet pre-trained model on CIFAR-100 with best learning rate...")
finetune_optimizer = optim.Adam([
    {'params': full_imagenet_pretrained_model.parameters(), 'lr': imagenet_best_lr / 10},
    {'params': full_imagenet_pretrained_model[-1].parameters(), 'lr': imagenet_best_lr}
])
imagenet_finetune_writer = SummaryWriter(log_dir='logs/imagenet_finetune')

for epoch in range(num_epochs_finetune):
    train_loss, train_acc, val_loss, val_acc = train_and_validate(train_loader_cifar100, test_loader_cifar100, full_imagenet_pretrained_model, criterion, finetune_optimizer, epoch, imagenet_finetune_writer)
    print(f'Epoch [{epoch+1}/{num_epochs_finetune}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
imagenet_finetune_writer.close()

# 保存 ImageNet 预训练并微调后的模型权重
torch.save(full_imagenet_pretrained_model.state_dict(), 'imagenet_finetuned.pth')

# 重新导入模型权重的语句
# full_imagenet_pretrained_model.load_state_dict(torch.load('imagenet_finetuned.pth'))

# 训练和评估随机初始化模型
print("Training randomly initialized model on CIFAR-100...")
random_init_model = get_resnet18_model(pretrained=False).cuda()
full_random_init_model = add_fc_layer(random_init_model, num_classes)

#可以在这里直接设置学习率
random_best_lr = 2e-3

# 使用最佳学习率训练和评估随机初始化模型
print("Training randomly initialized model on CIFAR-100 with best learning rate...")
finetune_optimizer = optim.Adam([
    {'params': full_random_init_model.parameters(), 'lr': random_best_lr / 10},
    {'params': full_random_init_model[-1].parameters(), 'lr': random_best_lr}
])
random_finetune_writer = SummaryWriter(log_dir='logs/random_finetune')

for epoch in range(num_epochs_finetune):
    train_loss, train_acc, val_loss, val_acc = train_and_validate(train_loader_cifar100, test_loader_cifar100, full_random_init_model, criterion, finetune_optimizer, epoch, random_finetune_writer)
    print(f'Epoch [{epoch+1}/{num_epochs_finetune}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
random_finetune_writer.close()

# 保存随机初始化并微调后的模型权重
torch.save(full_random_init_model.state_dict(), 'random_finetuned.pth')

# 重新导入模型权重的语句
# full_random_init_model.load_state_dict(torch.load('random_finetuned.pth'))

print("Experiment completed!")
