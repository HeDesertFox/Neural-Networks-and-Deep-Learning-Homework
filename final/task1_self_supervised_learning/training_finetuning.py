import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import get_resnet18_model
from data_preparation import get_dataloaders

def train_cifar100(train_loader, model, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, (images, targets) in enumerate(tqdm(train_loader)):
        images, targets = images.cuda(), targets.cuda()

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % 10 == 0:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate_cifar100(val_loader, model, criterion, epoch, writer):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader)):
            images, targets = images.cuda(), targets.cuda()

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    return avg_loss, accuracy

def adjust_learning_rate(optimizer, epoch, lr):
    if epoch < 30:
        lr = lr
    elif epoch < 60:
        lr = lr * 0.1
    else:
        lr = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def finetune(train_loader, val_loader, model, lr_list, num_epochs):
    criterion = nn.CrossEntropyLoss()
    best_lr = None
    best_acc = 0.0
    for lr in lr_list:
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': lr / 10},
            {'params': model.fc.parameters(), 'lr': lr}
        ])
        writer = SummaryWriter(log_dir=f'logs/lr_{lr}')
        for epoch in range(num_epochs):
            adjust_learning_rate(optimizer, epoch, lr)
            train_loss, train_acc = train_cifar100(train_loader, model, criterion, optimizer, epoch, writer)
            val_loss, val_acc = validate_cifar100(val_loader, model, criterion, epoch, writer)
            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr
        writer.close()
    return best_lr

if __name__ == "__main__":
    batch_size = 128
    num_epochs = 100
    lr_list = [0.1, 0.01, 0.001]

    train_loader_mini_imagenet, test_loader_mini_imagenet, train_loader_cifar100, test_loader_cifar100 = get_dataloaders(batch_size)

    # 微调和评估
    model = get_resnet18_model(pretrained=False, moco_pretrain=True, num_classes=100).cuda()
    best_lr = finetune(train_loader_cifar100, test_loader_cifar100, model, lr_list, num_epochs)
    print(f'Best learning rate for fine-tuning: {best_lr}')
