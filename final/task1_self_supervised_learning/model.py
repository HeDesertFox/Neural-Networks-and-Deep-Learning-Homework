import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from collections import OrderedDict

class ResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()  # 移除最后的全连接层

    def forward(self, x):
        features = self.model(x)
        return features

class MoCo(nn.Module):
    def __init__(self, base_encoder, feature_dim=128, K=65536, m=0.999, T=0.07):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # 创建动量编码器
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        # 创建一个MLP头（投影头）
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(self.encoder_q.feature_dim, self.encoder_q.feature_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_q.feature_dim, feature_dim)
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(self.encoder_k.feature_dim, self.encoder_k.feature_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_k.feature_dim, feature_dim)
        )

        # 队列和标签
        self.register_buffer("queue", torch.randn(feature_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # 动量编码器参数冻结
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # 不更新动量编码器参数

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        logits_q = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits_q /= self.T

        logits_k = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        logits_k /= self.T

        self._dequeue_and_enqueue(k)

        return logits_q, logits_k, q, k

def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output