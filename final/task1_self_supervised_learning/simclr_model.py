import torchvision.models as models
import torch.nn as nn
import torch


class SimCLR_ResNet18(nn.Module):
    def __init__(self, dims, *args, **kwargs) -> None:
        super(SimCLR_ResNet18, self).__init__(*args, **kwargs)
        self.backbone = models.resnet18(num_classes=dims, weights=None)
        dim_in = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(), self.backbone.fc
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class FullModel(nn.Module):
    def __init__(self, num_classes, pretrained, *args, **kwargs) -> None:
        super(FullModel, self).__init__(*args, **kwargs)
        if pretrained:
            self.backbone = models.resnet18(num_classes=num_classes, pretrained=True)
        else:
            self.backbone = models.resnet18(num_classes=num_classes, weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, self.backbone.fc.in_features),
            nn.ReLU(),
            self.backbone.fc,
        )

    def forward(self, x):
        return self.backbone(x)
