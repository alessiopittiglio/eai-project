import torch.nn as nn
from torchvision import models

# This class is a simplified version of the original BaseModel
# As a baseline model, it's very simple. Unlike the original version, 
# this one does not include temporal aggregation (e.g., average pooling).

class ResNet18SingleFrame(nn.Module):
    """
    Model based on ResNet18 for single frame classification.
    Input shape: (B, C, H, W)
    Output: Logits for 2 classes
    """
    def __init__(self, weights="ResNet18_Weights.DEFAULT", num_classes=2):
        super().__init__()
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
