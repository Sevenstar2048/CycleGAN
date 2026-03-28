import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Classifier(nn.Module):
    """Office-Home 分类器基线。"""

    def __init__(self, num_classes: int = 65):
        super().__init__()
        backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
