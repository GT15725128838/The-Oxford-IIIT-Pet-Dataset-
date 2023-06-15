import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from sklearn.preprocessing import LabelEncoder
from net import Net

# 加载模型权重
model = torch.load('./model_20230615160903.pth')
model.eval()

# # 加载标签编码器
# label_encoder = LabelEncoder()
# label_encoder.classes_ = np.load('./to/label_encoder_classes.npy')


def preprocess_image(image):
    # 图像预处理
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = ToTensor()(image)
    image = image.unsqueeze(0)
    return image


def predict_image(image):
    # 图像预测
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        # predicted_label = label_encoder.inverse_transform([predicted.item()])
        return predicted


# 读取新图像并进行预测
new_image = cv2.imread('./test_img/meiduan01.jpg')
predicted_label = predict_image(new_image)
print('Predicted label:', predicted_label)
