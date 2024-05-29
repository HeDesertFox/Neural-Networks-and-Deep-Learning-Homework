import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from model import ResNet18, MoCo
from data_preparation import get_mini_imagenet_data, get_cifar100_data

def train_moco(model, train_loader, criterion, optimizer, device, epoch, log_interval, writer):
    model.train()
    running_loss = 0.0
    for batch_idx, (im_q, im_k) in enumerate(train_loader):
        im_q, im_k = im_q.to(device), im_k.to(device)
        optimizer.zero_grad()
        logits_q, logits_k, q, k = model(im_q, im_k)
        labels = torch.arange(logits_q.size(0)).to(device)
        loss = criterion(logits_q, labels) + criterion(logits_k, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            writer.add_scalar('training_loss', running_loss / log_interval, epoch * len(train_loader) + batch_idx)
            print(f'Train Epoch: {epoch} [{batch_idx * len(im_q)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / log_interval:.6f}')
            running_loss = 0.0

def finetune_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epoch, log_interval, writer, phase='finetune'):
    if phase == 'finetune':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader if phase == 'finetune' else val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if phase == 'finetune':
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % log_interval == 0:
            writer.add_scalar(f'{phase}_loss', running_loss / log_interval, epoch * len(train_loader) + batch_idx)
            writer.add_scalar(f'{phase}_accuracy', 100. * correct / total, epoch * len(train_loader) + batch_idx)
            print(f'{phase.capitalize()} Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {running_loss / log_interval:.6f}\tAccuracy: {100. * correct / total:.2f}%')
            running_loss = 0.0
