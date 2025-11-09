import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic conv block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.pool(self.relu(self.bn(self.conv(x))))

class ResNet12(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, channels_mult=1.0):
        super().__init__()
        c = int(base_channels * channels_mult)
        self.layer1 = ConvBlock(input_channels, c)
        self.layer2 = ConvBlock(c, c*2)
        self.layer3 = ConvBlock(c*2, c*4)
        self.layer4 = ConvBlock(c*4, c*8)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = c*8
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x
