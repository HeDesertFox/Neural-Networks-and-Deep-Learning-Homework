import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import get_resnet18_model
from data_preparation import get_dataloaders

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # 创建 query encoder
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # 修改最后一层为MLP
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k)

        return logits, labels

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output

def train_moco(train_loader, model, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = 0.0
    for i, (images, _) in enumerate(tqdm(train_loader)):
        images = [img.cuda() for img in images]
        im_q, im_k = images

        logits, labels = model(im_q, im_k)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

    avg_loss = total_loss / len(train_loader)
    return avg_loss




if __name__ == "__main__":
    batch_size = 128
    num_epochs = 100

    train_loader_mini_imagenet, test_loader_mini_imagenet, train_loader_cifar100, test_loader_cifar100 = get_dataloaders(batch_size)

    # MoCo 预训练
    moco_model = MoCo(base_encoder=get_resnet18_model, dim=128, K=65536, m=0.999, T=0.07, mlp=False).cuda()
    moco_optimizer = optim.SGD(moco_model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4)
    moco_criterion = nn.CrossEntropyLoss().cuda()
    moco_writer = SummaryWriter(log_dir='logs/moco')
    for epoch in range(num_epochs):
        train_loss = train_moco(train_loader_mini_imagenet, moco_model, moco_criterion, moco_optimizer, epoch, moco_writer)
        print(f'Epoch [{epoch+1}/{num_epochs}], MoCo Loss: {train_loss:.4f}')
    moco_writer.close()
