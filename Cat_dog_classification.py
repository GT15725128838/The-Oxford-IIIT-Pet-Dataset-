import os
import cv2
import re
import datetime
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from net import Net

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize([256, 256]),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_images_and_labels(dataset_dir):
    """
    加载数据，预处理数据
    :param dataset_dir:
    :return:
    """
    print('-' * 50)
    print('Data loading...')
    images = []
    labels = []
    # 遍历图片
    # 猫
    cat_dir = dataset_dir + 'cat/'
    for file_name in sorted(os.listdir(cat_dir)):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(cat_dir, file_name)
            # 读取图片, RGB
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            labels.append('cat')
    # 狗
    dog_dir = dataset_dir + 'dog/'
    for file_name in sorted(os.listdir(dog_dir)):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(dog_dir, file_name)
            # 读取图片, RGB
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            labels.append('dog')
    print('Loading completed')
    print('-' * 50)
    return images, np.array(labels)


# 创建自定义数据集类
class PetDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = transform(image)
        return image, label


if __name__ == '__main__':
    # 图片路径
    dataset_dir = './data/images/'
    images, labels = load_images_and_labels(dataset_dir)
    # 标签编码
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    # 保存编码
    np.save('./cat_dog_classes.npy', label_encoder.classes_)
    print("Number of labels:", len(set(labels_encoded)))

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=10)

    # 调用类加载图像和标签
    train_dataset = PetDataset(X_train, y_train)
    test_dataset = PetDataset(X_test, y_test)
    print('-' * 50)
    print("Number of train_dataset:", len(train_dataset))
    print("Number of test_dataset:", len(test_dataset))

    # 调用网络模型
    model = Net(len(set(labels_encoded)))
    # 打印网络结构
    print(model)

    # 加载训练数据
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # 迭代次数
    num_epochs = 20
    # 判断是否存在cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 迭代训练
    print('-' * 50)
    print('Start of training')
    for epoch in range(num_epochs):
        # 损失函数
        train_loss = 0.0
        # 准确度
        train_acc = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 执行下一步更新
            optimizer.step()
            # 计算总loss（平均loss×样本数）
            train_loss += loss.item() * labels.size(0)
            # 准确度分析
            _, predicted = torch.max(outputs.data, 1)
            # 计算预测值与真实值相等的个数，也就是accuracy
            train_correct = (predicted == labels).sum()
            # 对每个batch的accuracy进行累加
            train_acc += train_correct.item()

        train_acc = 100 * train_acc / len(train_dataset)
        # 迭代完一次打印LOSS与ACC
        print('-' * 50)
        print('Epoch [{}/{}]: Loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(train_dataset)))
        print('Train Accuracy: {:.2f}%'.format(train_acc))

        # 测试集预测准确率
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # 准确度
        test_acc = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                # 前向传播
                outputs = model(images)
                # 准确度分析
                _, predicted = torch.max(outputs.data, 1)
                test_acc += (predicted == labels).sum().item()

        test_acc = 100 * test_acc / len(test_dataset)
        print('Test Accuracy: {:.2f}%'.format(test_acc))
        print('-' * 50)
    # 训练结束
    print('End of training')
    print('-' * 50)

    # 模型保存
    # 获取当前日期和时间
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # 添加时间戳到文件名中
    save_path = f'./model_{timestamp}.pth'
    # 保存整个模型
    torch.save(model, save_path)
