import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_model(pretrained=False, num_classes=100):
    """
    获取 ResNet-18 模型。

    参数:
    pretrained (bool): 是否加载 ImageNet 预训练权重。
    num_classes (int): 分类任务的类别数量。

    返回:
    nn.Module: ResNet-18 模型。
    """
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
