import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # BN层1
        self.bn1 = nn.BatchNorm2d(16)
        # Relu1
        self.relu1 = nn.ReLU()

        # 池化层1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # BN层2
        self.bn2 = nn.BatchNorm2d(32)
        # Relu2
        self.relu2 = nn.ReLU()

        # 池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层1
        self.fc1 = nn.Linear(32 * 64 * 64, 256)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.25)
        # 全连接层2
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.dropout(x)
        x = self.fc2(x)
        return x

