import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

def train_model(model, dataloaders, device, criterion, optimizer, num_epochs=25, writer=None):
    """
    训练模型并在训练和验证阶段跟踪性能。
    参数:
        model (torch.nn.Module): 要训练的模型。
        dataloaders (dict): 包含' train '和' val '数据的数据加载器。
        device (torch.device): 模型和数据应在哪个设备上运行。
        criterion (callable): 用于计算损失的函数。
        optimizer (torch.optim.Optimizer): 优化器。
        num_epochs (int): 训练的总轮次。
        writer (torch.utils.tensorboard.SummaryWriter, optional): 用于记录训练过程的TensorBoard writer。
    返回:
        torch.nn.Module: 训练后的模型。
    """
    model.to(device)
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            if writer:
                writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
                writer.add_scalar(f'{phase} Accuracy', epoch_acc, epoch)

    return model

def hyperparameter_tuning(model, dataloaders, device, num_epochs_list, lr_list, use_pretrained=True):
    """
    对模型进行超参数调整，以找到最佳的学习率和训练轮次。
    参数:
        model (torch.nn.Module): 模型原型。
        dataloaders (dict): 包含训练和验证数据的加载器。
        device (torch.device): 设备信息。
        num_epochs_list (list of int): 要尝试的训练轮次列表。
        lr_list (list of float): 要尝试的学习率列表。
        use_pretrained (bool): 模型是否使用预训练权重。
    返回:
        dict: 包含最佳学习率、轮次和验证准确率的字典。
    """
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_params = {'lr': None, 'num_epochs': None, 'accuracy': 0}

    for lr in lr_list:
        for num_epochs in num_epochs_list:
            print(f"Training with lr={lr}, num_epochs={num_epochs}")

            # Adjust learning rate based on whether the model is pretrained
            if use_pretrained:
                optimizer = optim.SGD([
                    {'params': model.fc.parameters(), 'lr': lr},
                    {'params': (p for n, p in model.named_parameters() if 'fc' not in n), 'lr': lr / 10}
                ], momentum=0.9)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

            model_copy = deepcopy(model).to(device)
            trained_model = train_model(model_copy, dataloaders, device, criterion, optimizer, num_epochs)
            trained_model.eval()

            val_corrects = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = trained_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data)
                    val_total += labels.size(0)

            final_val_acc = val_corrects.double() / val_total
            print(f"Final validation accuracy: {final_val_acc:.4f}")

            if final_val_acc > best_acc:
                best_acc = final_val_acc
                best_params = {'lr': lr, 'num_epochs': num_epochs, 'accuracy': final_val_acc}
                print(f"New best accuracy: {final_val_acc:.4f} with lr={lr} and num_epochs={num_epochs}")

    print(f"Best Params: lr={best_params['lr']}, num_epochs={best_params['num_epochs']}, Accuracy={best_params['accuracy']:.4f}")
    return best_params
