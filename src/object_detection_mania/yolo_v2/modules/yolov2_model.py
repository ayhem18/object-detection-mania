import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class ConvBlock(nn.Module):
    """
    A standard building block: Conv3x3 -> Norm -> Activation
    Includes an optional residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn(self.conv(x)))
        if self.use_residual:
            out = out + identity
        return out

class ConvHead(nn.Module):
    """
    The detection head consisting of multiple convolutional blocks.
    """
    def __init__(self, in_channels: int, num_blocks: int, hidden_channels: int = 1024):
        super().__init__()
        blocks = []
        # First block to transition to hidden_channels
        blocks.append(ConvBlock(in_channels, hidden_channels, use_residual=False))
        
        # Subsequent blocks with residual connections
        for _ in range(num_blocks - 1):
            blocks.append(ConvBlock(hidden_channels, hidden_channels, use_residual=True))
            
        self.feature_extractor = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class YoloV2(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 backbone_out_channels: int, 
                 num_anchors: int, 
                 num_classes: int, 
                 num_conv_blocks: int = 2):
        """
        YOLOv2 Model Architecture.
        
        Args:
            backbone (nn.Module): The feature extractor (e.g., Darknet, ResNet).
            backbone_out_channels (int): Number of output channels from the backbone.
            num_anchors (int): Number of anchors per grid cell.
            num_classes (int): Number of classes to detect.
            num_conv_blocks (int): Number of 3x3 conv blocks in the detection head.
        """
        super().__init__()
        self.backbone = backbone
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # 1. Detection Head (Feature extraction)
        # Standard hidden channels for YOLOv2 is 1024
        self.head = ConvHead(backbone_out_channels, num_conv_blocks, hidden_channels=1024)
        
        # 2. Prediction Layer (1x1 Conv)
        # Output channels: num_anchors * (5 + num_classes)
        # 5: tx, ty, tw, th, obj
        self.out_channels = num_anchors * (5 + num_classes)
        self.predictor = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image batch (B, 3, H, W)
        Returns:
            torch.Tensor: Raw predictions (B, num_anchors * (5 + num_classes), H_feat, W_feat)
        """
        # Feature extraction from backbone
        features = self.backbone(x)
        
        # Apply detection head
        head_features = self.head(features)
        
        # Final prediction
        out = self.predictor(head_features)
        
        return out
