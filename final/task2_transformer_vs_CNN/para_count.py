import torch
from torchvision import models
from vit_pytorch import ViT

# 定义打印模型层次架构和参数量的函数
def print_model_parameters(model):
    total_params = 0
    print(f"{'Layer':<40} {'Shape':<18} {'Params':<8}")
    print("="*66)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_shape = list(param.shape)
        param_num = param.numel()
        total_params += param_num
        print(f"{name:<40} {str(param_shape):<18} {param_num:<8}")
    print("="*66)
    print(f"Total Parameters: {total_params}")

# 定义ResNet-18模型
resnet18 = models.resnet18(num_classes=100)
print("ResNet-18 Architecture and Parameters:")
print_model_parameters(resnet18)

# 定义ViT模型
vit_model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=100,
    dim=192,
    depth=25,
    heads=6,
    mlp_dim=384,
    dropout=0.1,
    emb_dropout=0.1
)
print("\nViT Architecture and Parameters:")
print_model_parameters(vit_model)
