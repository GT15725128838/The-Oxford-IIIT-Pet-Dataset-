import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize([256, 256]),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 加载模型权重
model = torch.load('./model_20230626170102.pth')
model.eval()
# 加载标签编码器
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('./cat_dog_classes.npy')


# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    cv_show('Img', img)
    # 图像预处理
    image = Image.open(image_path).convert('RGB')  # 读取图像
    image = transform(image)  # 图像预处理
    # 拓展维度
    image = image.unsqueeze(0)
    return image


def predict_image(image_path):
    # 图像预测
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = label_encoder.inverse_transform([predicted.item()])
        return predicted_label


if __name__ == '__main__':
    # 读取新图像
    image_path = './test_img/beagle01.jpg'
    # 进行预测
    predicted_label = predict_image(image_path)
    print('Predicted label:', predicted_label)
