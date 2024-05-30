import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18MoCo(nn.Module):
    def __init__(self, pretrained=False, moco_pretrain=False, num_classes=100):
        super(ResNet18MoCo, self).__init__()
        # 加载 ResNet-18 模型
        if pretrained and not moco_pretrain:
            self.model = models.resnet18(pretrained=True)
        else:
            self.model = models.resnet18(pretrained=False)

        # 如果是 MoCo 预训练
        if moco_pretrain:
            self.model.fc = nn.Identity()  # MoCo 预训练不需要最后的分类层

        # 替换最后一层以适应 CIFAR-100 分类任务
        if not moco_pretrain:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_resnet18_model(pretrained=False, moco_pretrain=False, num_classes=100):
    """
    获取 ResNet-18 模型。

    Args:
        pretrained (bool): 是否加载 ImageNet 预训练权重。
        moco_pretrain (bool): 是否加载 MoCo 预训练权重。
        num_classes (int): 分类任务的类别数量。

    Returns:
        model (nn.Module): ResNet-18 模型。
    """
    model = ResNet18MoCo(pretrained=pretrained, moco_pretrain=moco_pretrain, num_classes=num_classes)
    return model





if __name__ == "__main__":
    # 测试随机初始化的模型
    model_random = get_resnet18_model(pretrained=False, moco_pretrain=False, num_classes=100)
    print(model_random)

    # 测试 ImageNet 预训练的模型
    model_imagenet = get_resnet18_model(pretrained=True, moco_pretrain=False, num_classes=100)
    print(model_imagenet)

    # 测试 MoCo 预训练的模型（此处假设已有 MoCo 预训练权重）
    model_moco = get_resnet18_model(pretrained=False, moco_pretrain=True, num_classes=100)
    print(model_moco)
