import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Classifier(nn.Module):
    """Office-Home 分类器基线。"""

    def __init__(self, num_classes: int = 65):
        super().__init__()
        backbone = resnet18(weights=None)
        self.feature_dim = backbone.fc.in_features
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.flatten(1)
        return feat

    def classify_features(self, feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classify_features(feat)
