import numpy as np
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import normalize
import cv2
import matplotlib.pyplot as plt

# 指定模型路径和图像路径
model_path = 'unet_bubble_detector.keras'
original_image_path = 'data/test/20210708143022.jpg'  # 替换为你的图像路径

# 加载模型
model = load_model(model_path)

# 读取图像（确保以彩色模式加载）
image = load_img(original_image_path, color_mode="rgb")  # 图像已经是640x480，不需要调整大小
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# 归一化图像数据; 确保这和模型训练时使用的方法一致
image = normalize(image, axis=-1)

# 使用模型进行预测
prediction = model.predict(image)
print("Prediction min value:", prediction.min())
print("Prediction max value:", prediction.max())

# 将预测结果转换为二值化图像
#threshold = 0.00250  # 您可以调整这个阈值
threshold = 0.5  # 您可以调整这个阈值
thresholded = (prediction > threshold).astype(np.uint8)
print("Thresholded unique values:", np.unique(thresholded))  # 应该包含0和1

segmented_image = thresholded[0, :, :, 0] * 255  # 将气泡部分转换为全白

# 保存分割后的图像
# 获取原图片所在的文件夹路径
dir_name = os.path.dirname(original_image_path)
# 创建分割后的图像文件名
segmented_image_filename = 'segmented_' + os.path.basename(original_image_path)
# 创建分割后图像的保存路径
segmented_image_path = os.path.join(dir_name, segmented_image_filename)
# 保存分割后的图像
cv2.imwrite(segmented_image_path, segmented_image)


