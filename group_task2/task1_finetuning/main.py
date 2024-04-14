import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from data_loading_preprocessing import load_data
from model import initialize_model
from training_fine_tuning import train_model, hyperparameter_tuning

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据加载
download_url = 'https://your-data-url.com/cub200_2011.zip'
train_loader, val_loader = load_data('.', batch_size=32, k=5, download_url=download_url)

# 创建包含训练和验证加载器的字典
dataloaders = {'train': train_loader, 'val': val_loader}

# 超参数列表
lr_list = [0.001, 0.0001, 0.00001]
num_epochs_list = [10, 20, 30]

# 对预训练模型进行超参数调优
print("Tuning hyperparameters for pretrained model...")
model, _ = initialize_model("resnet", 200, use_pretrained=True)
model = model.to(device)
best_params_pretrained = hyperparameter_tuning(model, dataloaders, device, num_epochs_list, lr_list, use_pretrained=True)
print(f"Best hyperparameters for pretrained model: {best_params_pretrained}")

# 对随机初始化模型进行超参数调优
print("Tuning hyperparameters for model with random initialization...")
model, _ = initialize_model("resnet", 200, use_pretrained=False)
model = model.to(device)
best_params_random = hyperparameter_tuning(model, dataloaders, device, num_epochs_list, lr_list, use_pretrained=False)
print(f"Best hyperparameters for randomly initialized model: {best_params_random}")

# 准备TensorBoard
writer_pretrained = SummaryWriter('runs/pretrained_model')
writer_random = SummaryWriter('runs/random_init_model')

# Re-train the pretrained model
model_pretrained, _ = initialize_model("resnet", 200, use_pretrained=True)
model_pretrained = model_pretrained.to(device)

# 设置不同的学习率
optimizer_pretrained = optim.SGD([
    {'params': model_pretrained.fc.parameters(), 'lr': best_params_pretrained['lr'] * 10},  # 最后一层
    {'params': (p for n, p in model_pretrained.named_parameters() if 'fc' not in n), 'lr': best_params_pretrained['lr']}  # 其他所有层
], momentum=0.9)

criterion = nn.CrossEntropyLoss()
trained_model_pretrained = train_model(
    model_pretrained, dataloaders, device, criterion, optimizer_pretrained, num_epochs=best_params_pretrained['num_epochs'], writer=writer_pretrained
)
writer_pretrained.close()  # 关闭TensorBoard writer

# Re-train the model from random initialization
model_random, _ = initialize_model("resnet", 200, use_pretrained=False)
model_random = model_random.to(device)

# 设置优化器
optimizer_random = optim.SGD(model_random.parameters(), lr=best_params_random['lr'], momentum=0.9)

trained_model_random = train_model(
    model_random, dataloaders, device, criterion, optimizer_random, num_epochs=best_params_random['num_epochs'], writer=writer_random
)
writer_random.close()  # 关闭TensorBoard writer
