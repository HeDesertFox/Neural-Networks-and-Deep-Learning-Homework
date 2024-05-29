from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights, AlexNet_Weights

def initialize_model(model_name, num_classes, use_pretrained=True):
    """
    初始化一个给定类别数量的模型，并可选择是否使用预训练权重。
    参数:
        model_name (str): 要初始化的模型名称。
        num_classes (int): 输出类别的数量。
        use_pretrained (bool): 是否使用预训练的模型权重。
    返回:
        model (torch.nn.Module): 初始化后的模型。
        input_size (int): 模型需要的输入图像尺寸。
    """

    model = None  # 初始化模型变量
    input_size = 0  # 初始化输入尺寸变量
    weights = None  # 初始化权重变量

    if model_name == "resnet":
        # 使用ResNet18模型
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features  # 获取全连接层前的特征数量
        model.fc = nn.Linear(num_ftrs, num_classes)  # 替换全连接层以适应目标类别数
        input_size = 224  # ResNet18需要的输入尺寸为224x224

    elif model_name == "alexnet":
        # 使用AlexNet模型
        weights = AlexNet_Weights.DEFAULT if use_pretrained else None
        model = models.alexnet(weights=weights)
        num_ftrs = model.classifier[6].in_features  # 获取分类层前的特征数量
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)  # 替换分类层以适应目标类别数
        input_size = 224  # AlexNet需要的输入尺寸为224x224

    else:
        print("Invalid model name, exiting...")  # 如果模型名称无效，则打印错误信息并退出
        exit()

    return model, input_size  # 返回初始化的模型和输入尺寸
