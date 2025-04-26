import torch
import torch.nn as nn
from torchvision import models, transforms

class Model(nn.Module):
    def __init__(self, weights="ResNet18_Weights.DEFAULT"):
        super(Model, self).__init__()
        self.backbone = models.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        # x shape: (B, num_frames, C, H, W)
        B, F, C, H, W = x.size()
        x = x.view(B * F, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, F, -1).mean(1)  # average pooling over frames
        return feats
    
def get_loss_fn():
    return nn.CrossEntropyLoss()

def get_optimizer(model_params):
    return torch.optim.SGD(model_params, lr=0.001, momentum=0.9)

def get_data_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])