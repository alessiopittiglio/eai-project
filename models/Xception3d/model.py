import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

#Xception model for video classification
#3d version of https://arxiv.org/abs/1610.02357

class SeparableConv3d(nn.Module):
    """
    Depthwise separable 3D convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(SeparableConv3d, self).__init__()
        # Depthwise: groups=in_channels
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        # Pointwise
        self.pointwise = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class XceptionBlock3D(nn.Module):
    """
    Single Xception block (entry, middle or exit flow)
    """
    def __init__(self, in_channels, out_channels, reps, stride=1, grow_first=True):
        super(XceptionBlock3D, self).__init__()
        filters = out_channels
        self.skip = None
        if in_channels != out_channels or stride != 1:
            # projection for skip connection
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        layers = []
        # first conv in block
        if grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv3d(in_channels, filters,
                                          kernel_size=3, stride=1, padding=1, bias=False))
            in_channels = filters
        for _ in range(reps - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv3d(in_channels, in_channels,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        # last conv: change resolution if stride>1
        if not grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv3d(in_channels, filters,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        if stride != 1:
            layers.append(nn.MaxPool3d(kernel_size=3, stride=stride, padding=1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        res = x
        x = self.conv(x)
        if self.skip is not None:
            res = self.skip(res)
        x += res
        return x


class Xception3D(nn.Module):
    """
    3D Xception backbone adapted for video input
    Input: (B, C=3, T, H, W) 
    # Example input: 2 videos, 3 channels, 16 frames, 224x224 resolution
    Output: feature map (B, 2048, T', H', W')
    """
    def __init__(self):
        super(Xception3D, self).__init__()
        # Entry flow
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, stride=(1,2,2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)

        self.block1 = XceptionBlock3D(64, 128, reps=3, stride=2, grow_first=True)
        self.block2 = XceptionBlock3D(128, 256, reps=3, stride=2, grow_first=True)
        self.block3 = XceptionBlock3D(256, 728, reps=3, stride=2, grow_first=True)

        # Middle flow: 8 blocks
        self.middle_blocks = nn.Sequential(
            *[XceptionBlock3D(728, 728, reps=3, stride=1, grow_first=True) for _ in range(8)]
        )

        # Exit flow
        self.block_exit = XceptionBlock3D(728, 1024, reps=3, stride=2, grow_first=False)
        self.conv3 = SeparableConv3d(1024, 1536, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = SeparableConv3d(1536, 2048, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # Entry
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Middle
        x = self.middle_blocks(x)
        # Exit
        x = self.block_exit(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class Model(nn.Module):
    """
    Example video detection head using Xception3D backbone.
    Performs classification over num_classes (could be detection logits).
    """
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        self.backbone = Xception3D()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc_out = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x: (B, 3, T, H, W)
        feat = self.backbone(x)              # (B, 2048, T', H', W')
        pooled = self.pool(feat).view(x.size(0), -1)  # (B, 2048)
        out = self.fc1(pooled)                # (B, 1024)
        out = self.fc_out(out)                # (B, num_classes)
        out = F.softmax(out, dim=1)          # (B, num_classes)
        return out
    
def get_loss_fn():
    return nn.CrossEntropyLoss()

def get_optimizer(model_params):
    optimizer = torch.optim.SGD(model_params, lr=0.045, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)
    return optimizer, scheduler

def get_data_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])