#!env
# Author
# Create Time: 2022/7/4
# License: None
"""
Info
"""

# import
import torch
from torch import nn
import torch.nn.functional as F

# model for LeNet

class LeNet(nn.Module):
    def __init__(self, cnt_1=20, cnt_2=50):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cnt_1, kernel_size=(5, 5))
        self.act1 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv2 = nn.Conv2d(in_channels=cnt_1, out_channels=cnt_2, kernel_size=(5, 5))
        self.act2 = nn.ReLU()
        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.fc1 = nn.Linear(cnt_2*4*4, 120)
        self.fc_act1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.fc_act2 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pooling1(self.conv1(x))
        x = self.pooling2(self.conv2(x))

        x = torch.flatten(x, 1)

        x = self.fc_act1(self.fc1(x))
        x = self.fc_act2(self.fc2(x))
        y = self.fc3(x)
        return y # prediction

if __name__ == '__main__':
    test_img = torch.rand((64, 1, 28, 28))
    model = LeNet(cnt_1=10, cnt_2=20)
    print(model(test_img))
