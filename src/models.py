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

# 3D Xception model for video classification
# Inspired by the original 2D Xception paper: https://arxiv.org/abs/1610.02357
    
class SeparableConv3d(nn.Module):
    """
    3D separable convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()

        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=1, bias=bias)
        
        # Removed BN from here, as it is applied 
        # after the activation function
        # in the Xception block
        # self.bn = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        # x = self.bn(x)
        return x
    
class XceptionBlock3D(nn.Module):
    """
    Single Xception block (entry, middle or exit flow)
    """
    def __init__(self, in_channels, out_channels, reps, stride=1, grow_first=True):
        super().__init__()
        filters = out_channels
        self.skip_connection = None # Renamed for clarity
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True) # Changed to inplace=True for efficiency

        layers = []
        current_channels = in_channels
        # First conv in block
        if grow_first:
            layers.append(self.relu)
            layers.append(SeparableConv3d(current_channels, filters, kernel_size=3, padding='same', bias=False)) 
            layers.append(nn.BatchNorm3d(filters))
            current_channels = filters

        for _ in range(reps - 1):
            layers.append(self.relu)
            layers.append(SeparableConv3d(current_channels, current_channels, kernel_size=3, padding='same', bias=False))
            layers.append(nn.BatchNorm3d(current_channels))

        if not grow_first:
            layers.append(self.relu)
            layers.append(SeparableConv3d(current_channels, filters, kernel_size=3, padding='same', bias=False))
            layers.append(nn.BatchNorm3d(filters))
            current_channels = filters

        if stride != 1:
            layers.append(nn.MaxPool3d(kernel_size=3, stride=stride, padding=1))

        self.conv_layers = nn.Sequential(*layers) # Renamed for clarity

    def forward(self, x):
        residual = x
        x = self.conv_layers(x)
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
        x += residual
        return x
    
class Xception3DBackbone(nn.Module):
    """
    3D Xception backbone. 
    Input: (B, C, T, H, W) 
    Output: (B, 2048, T', H', W')
    """
    def __init__(self, input_channels=3):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # Xception blocks as per the original paper
        self.block1 = XceptionBlock3D(64, 128, reps=2, stride=2, grow_first=True) # Changed reps to 2
        self.block2 = XceptionBlock3D(128, 256, reps=2, stride=2, grow_first=True)
        self.block3 = XceptionBlock3D(256, 728, reps=2, stride=2, grow_first=True)

        # Middle flow
        self.middle_blocks = nn.Sequential(
            *[XceptionBlock3D(728, 728, reps=3, stride=1, grow_first=True) for _ in range(4)] # Changed to 4 blocks
        )

        # Exit flow
        self.block_exit = XceptionBlock3D(728, 1024, reps=2, stride=2, grow_first=True) # Changed reps to 2
        self.conv3 = SeparableConv3d(1024, 1536, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm3d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = SeparableConv3d(1536, 2048, kernel_size=3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm3d(2048)
        self.relu4 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Assuming x is of shape (B, C, T, H, W)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Middle flow
        x = self.middle_blocks(x)
        # Exit flow
        x = self.block_exit(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        return x
    
class Xception3DClassifier(nn.Module):
    """
    Classifier using the Xception3D backbone.
    Input: (B, C, T, H, W)
    Output: (B, 1) logit
    """
    def __init__(self, input_channels=3):
        super().__init__()
        self.backbone = Xception3DBackbone(input_channels=input_channels)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, 1)
    
    def forward(self, x):
        features = self.backbone(x)                # (B, 2048, T', H', W')
        pooled = self.pool(features)               # (B, 2048, 1, 1, 1)
        pooled_flat = pooled.view(x.size(0), -1)   # (B, 2048)
        logits = self.fc(pooled_flat)              # (B, 1)
        return logits
