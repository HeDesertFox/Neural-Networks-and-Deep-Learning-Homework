import torch
import torch.nn as nn
import torch.optim as optim
import copy

def train_and_validate(train_loader, val_loader, model, criterion, optimizer, epoch):
    """
    训练和验证 CIFAR-100 模型。

    参数:
    train_loader (DataLoader): 训练数据加载器。
    val_loader (DataLoader): 验证数据加载器。
    model (nn.Module): 待训练和验证的模型。
    criterion (nn.Module): 损失函数。
    optimizer (Optimizer): 优化器。
    epoch (int): 当前训练周期。

    返回:
    train_avg_loss (float): 平均训练损失。
    train_accuracy (float): 训练准确率。
    val_avg_loss (float): 平均验证损失。
    val_accuracy (float): 验证准确率。
    """
    # 训练阶段
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, (images, targets) in enumerate(train_loader):
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

    train_avg_loss = total_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    # 验证阶段
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images, targets = images.cuda(), targets.cuda()

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_avg_loss = total_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    return train_avg_loss, train_accuracy, val_avg_loss, val_accuracy

def finetune(train_loader, val_loader, model, lr_list, num_epochs):
    """
    微调模型。

    参数:
    train_loader (DataLoader): 训练数据加载器。
    val_loader (DataLoader): 验证数据加载器。
    model (nn.Module): 待微调模型。
    lr_list (list of float): 学习率列表。
    num_epochs (int): 训练周期数。

    返回:
    best_lr (float): 最佳学习率。
    """
    criterion = nn.CrossEntropyLoss()
    best_lr = None
    best_acc = 0.0
    initial_state_dict = copy.deepcopy(model.state_dict())  # 保存初始模型权重

    for lr in lr_list:
        model_copy = copy.deepcopy(model)  # 创建模型的副本
        model_copy.load_state_dict(initial_state_dict)  # 加载初始模型权重
        optimizer = optim.Adam([
            {'params': model_copy[:-1].parameters(), 'lr': lr / 10},
            {'params': model_copy[-1].parameters(), 'lr': lr}
        ])
        for epoch in range(num_epochs):
            train_loss, train_acc, val_loss, val_acc = train_and_validate(train_loader, val_loader, model_copy, criterion, optimizer, epoch)
            print(f'Epoch [{epoch+1}/{num_epochs}], LR: {lr}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            if val_acc > best_acc:
                best_acc = val_acc
                best_lr = lr

    return best_lr
