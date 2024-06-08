import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
from model import get_resnet18_model

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        初始化 MoCo 模型。

        参数:
        base_encoder (nn.Module): 基础编码器（如 ResNet-18）。
        dim (int): 输出特征维度。
        K (int): 队列大小。
        m (float): 动量系数。
        T (float): 温度系数。
        mlp (bool): 是否使用 MLP 头部。
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # 创建 query encoder 和 key encoder
        self.encoder_q = base_encoder(pretrained=False)
        self.encoder_k = base_encoder(pretrained=False)

        # 获取原始 fc 层的输入特征维度
        dim_mlp = 512  # ResNet18 最后一个层的输出特征维度是 512

        if mlp:  # 如果使用 MLP 头部
            self.encoder_q.model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        else:
            self.encoder_q.model.fc = nn.Linear(dim_mlp, dim)
            self.encoder_k.model.fc = nn.Linear(dim_mlp, dim)

        # 初始化 key encoder 的参数与 query encoder 相同，并且不更新梯度
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # 初始化队列和队列指针
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        使用动量更新 key encoder 的参数。
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        将新的 keys 插入队列，并移出旧的 keys。

        参数:
        keys (torch.Tensor): 新的 keys。
        """
        if torch.distributed.is_initialized():
            keys = concat_all_gather(keys)  # 聚合所有进程的 keys
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)  # 当前队列指针

        if ptr + batch_size <= self.K:
            # 插入 keys 到队列
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 如果插入超出队列末端，需要将 keys 拆分并分段插入
            first_part_size = self.K - ptr
            self.queue[:, ptr:] = keys[:first_part_size].T
            self.queue[:, :batch_size - first_part_size] = keys[first_part_size:].T

        ptr = (ptr + batch_size) % self.K  # 更新队列指针

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        前向传播函数。

        参数:
        im_q (torch.Tensor): query 图像。
        im_k (torch.Tensor): key 图像。

        返回:
        logits (torch.Tensor): 分类 logits。
        labels (torch.Tensor): 分类标签。
        """
        q = self.encoder_q(im_q)  # 通过 query encoder 处理 query 图像
        q = nn.functional.normalize(q, dim=1)  # 对输出进行归一化

        with torch.no_grad():  # 在不计算梯度的情况下处理 key 图像
            self._momentum_update_key_encoder()  # 更新 key encoder
            k = self.encoder_k(im_k)  # 通过 key encoder 处理 key 图像
            k = nn.functional.normalize(k, dim=1)  # 对输出进行归一化

        # 计算正样本对和负样本对的 logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T  # 应用温度系数

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # 创建标签，正样本为0

        self._dequeue_and_enqueue(k)  # 更新队列

        return logits, labels

@torch.no_grad()
def concat_all_gather(tensor):
    """
    聚合所有进程的张量。

    参数:
    tensor (torch.Tensor): 输入张量。

    返回:
    torch.Tensor: 聚合后的张量。
    """
    if torch.distributed.is_initialized():
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor

def train_moco(train_loader, model, criterion, optimizer, epoch, writer=None):
    """
    训练 MoCo 模型。

    参数:
    train_loader (DataLoader): 训练数据加载器。
    model (nn.Module): MoCo 模型。
    criterion (nn.Module): 损失函数。
    optimizer (Optimizer): 优化器。
    epoch (int): 当前训练周期。
    writer (SummaryWriter, optional): TensorBoard 写入器。

    返回:
    avg_loss (float): 平均训练损失。
    """
    model.train()  # 设定模型为训练模式
    total_loss = 0.0
    to_pil = ToPILImage()
    to_tensor = ToTensor()
    for i, (images, _) in enumerate(tqdm(train_loader)):
        # 将图像加载到 GPU
        images = images.cuda()

        # 数据增强并转移到 GPU
        im_q = torch.stack([aug(to_pil(img)) for img in images]).cuda()
        im_k = torch.stack([aug(to_pil(img)) for img in images]).cuda()

        logits, labels = model(im_q, im_k)  # 前向传播计算 logits 和 labels
        loss = criterion(logits, labels)  # 计算损失

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        total_loss += loss.item()
        if writer and i % 25 == 0:
            writer.add_scalar('Instance discrimination task loss', loss.item(), epoch * len(train_loader) + i)  # 记录训练损失

    avg_loss = total_loss / len(train_loader)  # 计算平均损失
    return avg_loss

def aug(image):
    """
    数据增强函数。

    参数:
    image (PIL.Image): 输入图像。

    返回:
    torch.Tensor: 增强后的图像。
    """
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),  # 调整了 scale 参数范围
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),  # 新增 GaussianBlur 数据增强
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)
