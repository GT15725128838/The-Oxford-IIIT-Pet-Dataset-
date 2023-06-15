import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(224, 32, 3)
        # 添加更多的卷积层和池化层
        # ...
        self.fc1 = nn.Linear(227328, 64)
        self.fc2 = nn.Linear(64, 35)

    def forward(self, x):
        x = self.conv1(x)
        # 添加更多的卷积层和池化层
        # ...
        x = x.view(-1, 227328)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Net()
