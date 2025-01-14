from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from convolution2d import CustomConv2d
from dropout import CustomDropout
from relu import CustomReLU
class Net1(nn.Module):
    def __init__(self, drop=0.01):
        super(Net1, self).__init__()
        
        
        self.convblock = nn.Sequential(
            CustomConv2d(in_channels=3, out_channels=3, kernel_size=3, bias=False),
        )

    def forward(self, x):
        print(x.shape)
        x = self.convblock(x) #3
        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 5, 5)  # (batch_size, in_channels, height, width)
    model = Net1()
    x = model(x)
    print(x.shape)
    print(x)
    dropout = CustomDropout(p=0.5)
    x = dropout(x)
    print(x.shape)
    print(x)
    x = CustomReLU()(x)
    print(x.shape)
    print(x)
    
    # Initialize your custom convolution
    # custom_conv = CustomConv2d(3, 16, 3)
    # custom_output = custom_conv(x)
    # print(custom_output.shape)
