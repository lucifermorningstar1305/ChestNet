"""
@author: Adityam Ghosh
Date: 10-16-2023 

"""

from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlocks(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int|Tuple, stride: int|Tuple, padding: int):

        super().__init__()

        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.act = nn.ReLU(inplace=True)

        self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = x
        if self.stride != 1 or x.size() != out.size():
            
            identity = self.downsample(x)
        
        out += identity
        out = self.act(out)

        return out

class Classifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.model = nn.Sequential()

        self.model.add_module("conv1", nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )) # (56 x 56 x 64)
        
        self.model.add_module("layer_1", ResidualBlocks(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)) # (56 x 56 x 64)
        self.model.add_module("layer_2", ResidualBlocks(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)) # (56 x 56 x 64)
        self.model.add_module("layer_3", ResidualBlocks(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)) # (28 x 28 x 128)
        self.model.add_module("layer_4", ResidualBlocks(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)) # (28 x 28 x 128)
        self.model.add_module("layer_5", ResidualBlocks(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)) # (14 x 14 x 256)
        self.model.add_module("layer_6", ResidualBlocks(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)) # (14 x 14 x 256)
        self.model.add_module("layer_7", ResidualBlocks(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)) # (7 x 7 x 512)
        self.model.add_module("layer_8", ResidualBlocks(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)) # (7 x 7 x 512)

        self.model.add_module("avg_pool", nn.AdaptiveAvgPool2d(output_size=(1, 1))) # (1 x 1 x 512)
        self.model.add_module("fc", nn.Linear(in_features=512, out_features=n_classes)) 

        self.flatten = nn.Flatten()

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.size(2) == 224 and x.size(3) == 224, f"Expected size of the image to be (batch_size, 1, 224, 224). Found {x.size()}"

        out = self.model.conv1(x)
        out = self.model.layer_1(out)
        out = self.model.layer_2(out)
        out = self.model.layer_3(out)
        out = self.model.layer_4(out)
        out = self.model.layer_5(out)
        out = self.model.layer_6(out)
        out = self.model.layer_7(out)
        out = self.model.layer_8(out)

        out = self.model.avg_pool(out)
        out = self.flatten(out)
        out = self.model.fc(out)

        return out
    


class Similarifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential()

        self.model.add_module("conv1", nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )) # (56 x 56 x 64)
        
        self.model.add_module("layer_1", ResidualBlocks(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)) # (56 x 56 x 64)
        self.model.add_module("layer_2", ResidualBlocks(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)) # (56 x 56 x 64)
        self.model.add_module("layer_3", ResidualBlocks(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)) # (28 x 28 x 128)
        self.model.add_module("layer_4", ResidualBlocks(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)) # (28 x 28 x 128)
        self.model.add_module("layer_5", ResidualBlocks(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)) # (14 x 14 x 256)
        self.model.add_module("layer_6", ResidualBlocks(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)) # (14 x 14 x 256)
        self.model.add_module("layer_7", ResidualBlocks(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)) # (7 x 7 x 512)
        self.model.add_module("layer_8", ResidualBlocks(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)) # (7 x 7 x 512)

        self.model.add_module("avg_pool", nn.AdaptiveAvgPool2d(output_size=(1, 1))) # (1 x 1 x 512)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.size(2) == 224 and x.size(3) == 224, f"Expected size of the image to be (batch_size, 1, 224, 224). Found {x.size()}"

        out = self.model.conv1(x)
        out = self.model.layer_1(out)
        out = self.model.layer_2(out)
        out = self.model.layer_3(out)
        out = self.model.layer_4(out)
        out = self.model.layer_5(out)
        out = self.model.layer_6(out)
        out = self.model.layer_7(out)
        out = self.model.layer_8(out)

        out = self.model.avg_pool(out)
        out = out.flatten()
        
        return out
    