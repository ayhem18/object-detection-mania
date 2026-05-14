import torch
import torch.nn as nn
from typing import List, Tuple

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
import torch
import torch.nn as nn
from typing import List, Tuple

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