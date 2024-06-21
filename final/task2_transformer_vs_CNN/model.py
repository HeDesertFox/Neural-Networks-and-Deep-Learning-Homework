import torch
import torch.nn as nn
import torchvision.models as models
from vit_pytorch import ViT

# 定义ResNet模型
def get_resnet(num_classes=100, variant='resnet18'):
    if variant == 'resnet18':
        model = models.resnet18(pretrained=False, num_classes=num_classes)
    elif variant == 'resnet34':
        model = models.resnet34(pretrained=False, num_classes=num_classes)
    elif variant == 'resnet50':
        model = models.resnet50(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError("Unsupported ResNet variant")
    return model

# 定义ViT模型
def get_vit(image_size=32, patch_size=4, num_classes=100, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1, emb_dropout=0.1):
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout
    )
    return model
