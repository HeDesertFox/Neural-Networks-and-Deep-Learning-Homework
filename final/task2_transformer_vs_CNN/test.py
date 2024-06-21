import torch
from torch.utils.tensorboard import SummaryWriter
from data_preparation import get_data_loaders
from training import train, tune_hyperparameters
from model import get_resnet, get_vit  # 导入模型构造函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 100
alpha = 1.0  # CutMix alpha

# 获取数据加载器
train_loader, test_loader = get_data_loaders(batch_size)

# 定义学习率和权重衰减的取值范围
learning_rates = [0.001, 0.0001, 0.00001]
weight_decays = [0.0001, 0.001, 0.01]

# 准备ResNet模型并统计参数量
resnet_model = get_resnet(num_classes=100, variant='resnet18').to(device)
resnet_params = sum(p.numel() for p in resnet_model.parameters())
print(f'ResNet-18 model parameters: {resnet_params}')

# 超参数调优
print("Tuning ResNet hyperparameters...")
best_resnet_params = tune_hyperparameters(lambda: get_resnet(num_classes=100, variant='resnet18'), train_loader, test_loader, epochs, learning_rates, weight_decays, device, alpha)

# 用最优参数重新训练ResNet模型并用Tensorboard可视化
best_lr_resnet, best_wd_resnet = best_resnet_params
writer_resnet = SummaryWriter(log_dir='runs/ResNet_best')
resnet_model = get_resnet(num_classes=100, variant='resnet18').to(device)
optimizer_resnet = torch.optim.Adam(resnet_model.parameters(), lr=best_lr_resnet, weight_decay=best_wd_resnet)
criterion = torch.nn.CrossEntropyLoss()
train(resnet_model, train_loader, test_loader, optimizer_resnet, criterion, epochs, device, alpha, writer_resnet)
writer_resnet.close()

# 准备ViT模型并统计参数量
vit_model = get_vit(image_size=32, patch_size=4, num_classes=100, dim=192, depth=12, heads=3, mlp_dim=384, dropout=0.1, emb_dropout=0.1).to(device)
vit_params = sum(p.numel() for p in vit_model.parameters())
print(f'ViT model parameters: {vit_params}')

# 超参数调优
print("Tuning ViT hyperparameters...")
best_vit_params = tune_hyperparameters(lambda: get_vit(image_size=32, patch_size=4, num_classes=100, dim=192, depth=12, heads=3, mlp_dim=384, dropout=0.1, emb_dropout=0.1), train_loader, test_loader, epochs, learning_rates, weight_decays, device, alpha)

# 用最优参数重新训练ViT模型并用Tensorboard可视化
best_lr_vit, best_wd_vit = best_vit_params
writer_vit = SummaryWriter(log_dir='runs/ViT_best')
vit_model = get_vit(image_size=32, patch_size=4, num_classes=100, dim=192, depth=12, heads=3, mlp_dim=384, dropout=0.1, emb_dropout=0.1).to(device)
optimizer_vit = torch.optim.Adam(vit_model.parameters(), lr=best_lr_vit, weight_decay=best_wd_vit)
train(vit_model, train_loader, test_loader, optimizer_vit, criterion, epochs, device, alpha, writer_vit)
writer_vit.close()
