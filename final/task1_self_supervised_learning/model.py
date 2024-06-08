import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18MoCo(nn.Module):
    def __init__(self, pretrained=False):
        """
        初始化 ResNet-18 MoCo 模型。

        参数:
        pretrained (bool): 是否加载 ImageNet 预训练权重。
        """
        super(ResNet18MoCo, self).__init__()
        # 加载 ResNet-18 模型，如果是 ImageNet 预训练
        self.model = models.resnet18(pretrained=pretrained)
        # 去掉最后的分类层
        self.model.fc = nn.Identity()

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 输出张量。
        """
        return self.model(x)

def get_resnet18_model(pretrained=False):
    """
    获取 ResNet-18 模型。

    参数:
    pretrained (bool): 是否加载 ImageNet 预训练权重。

    返回:
    nn.Module: ResNet-18 模型。
    """
    model = ResNet18MoCo(pretrained=pretrained)
    return model

def add_fc_layer(model, num_classes):
    """
    为模型添加全连接层。

    参数:
    model (nn.Module): 基础模型。
    num_classes (int): 分类任务的类别数量。

    返回:
    nn.Module: 添加全连接层后的模型。
    """
    new_fc = nn.Linear(512, num_classes)
    full_model = nn.Sequential(
        model,
        new_fc
    )
    return full_model.cuda()
