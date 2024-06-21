import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy

def train(model, train_loader, test_loader, optimizer, criterion, epochs, device, writer=None):
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)

        avg_loss = running_loss / len(train_loader)
        accuracy_train = 100. * correct_train / total_train
        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy_train, epoch)

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                test_loss += criterion(outputs, targets).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        if writer:
            writer.add_scalar('Loss/val', test_loss, epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)

        print(f'Epoch [{epoch}/{epochs}] Train Loss: {avg_loss:.4f}, Accuracy: {accuracy_train:.2f}%, Validation Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

def tune_hyperparameters(model, train_loader, test_loader, epochs, lr_values, weight_decay_values, device, finetune=False):
    best_acc = 0
    best_hyperparams = None

    for lr in lr_values:
        for wd in weight_decay_values:
            model_copy = copy.deepcopy(model).to(device)
            criterion = nn.CrossEntropyLoss()

            if finetune:
                # 如果是微调，优化器对最后一层使用指定学习率，对其他层使用学习率的1/10
                optimizer = optim.Adam([
                    {'params': model_copy.fc.parameters(), 'lr': lr},
                    {'params': [param for name, param in model_copy.named_parameters() if name != 'fc'], 'lr': lr / 10}
                ], weight_decay=wd)
            else:
                optimizer = optim.Adam(model_copy.parameters(), lr=lr, weight_decay=wd)

            print(f'Tuning with learning rate: {lr}, weight decay: {wd}, finetune: {finetune}')
            train(model_copy, train_loader, test_loader, optimizer, criterion, epochs, device)

            # 验证集准确率计算
            model_copy.eval()
            correct = 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model_copy(data)
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(targets.view_as(pred)).sum().item()

            accuracy = 100. * correct / len(test_loader.dataset)
            print(f'Tuning Validation Accuracy: {accuracy:.2f}%')

            if accuracy > best_acc:
                best_acc = accuracy
                best_hyperparams = (lr, wd)

    print(f'Best Hyperparameters - Learning Rate: {best_hyperparams[0]}, Weight Decay: {best_hyperparams[1]} with Accuracy: {best_acc:.2f}%')
    return best_hyperparams
