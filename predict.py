import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from sklearn.preprocessing import LabelEncoder

# 加载模型权重
model = torch.load('./model_20230617173642.pth')
model.eval()
# 加载标签编码器
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('./label_encoder_classes.npy')


# 绘图展示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess_image(image):
    # 图像预处理
    image = cv2.resize(image, (256, 256))
    cv_show("Image", image)
    image = ToTensor()(image)
    image = image.unsqueeze(0)
    return image


def predict_image(image):
    # 图像预测
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = label_encoder.inverse_transform([predicted.item()])
        return predicted_label


if __name__ == '__main__':
    # 读取新图像
    new_image = cv2.imread('./test_img/american_pit_bull_terrier01.jpg')

    # 进行预测
    predicted_label = predict_image(new_image)
    print('Predicted label:', predicted_label)
