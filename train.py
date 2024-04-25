import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image

# 配置参数
img_width = 640  # 或者根据你的数据集调整
img_height = 480  # 或者根据你的数据集调整
num_channels = 3  # 原图的通道数，彩色图为3
num_mask_channels = 1  # 掩码的通道数，黑白掩码为1


# 构建U-Net模型
def unet(input_size=(480, 640, 3)):
    inputs = Input(input_size)

    # Contracting Path (Encoder)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Expansive Path (Decoder)
    up6 = UpSampling2D(size=(2, 2))(drop5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = Concatenate()([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    #conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 数据加载函数
def load_data(data_path):
    images = []
    masks = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.jpg'):
                    img_path = os.path.join(folder_path, file)
                    mask_path = os.path.splitext(img_path)[0] + '.png'

                    img = Image.open(img_path).resize((img_width, img_height))
                    img = np.array(img) / 255.0

                    mask = Image.open(mask_path).resize((img_width, img_height))
                    mask = np.array(mask) / 255.0
                    mask = mask[:, :, np.newaxis]  # 增加一个通道维度

                    images.append(img)
                    masks.append(mask)

    return np.array(images), np.array(masks)


# 加载数据
train_data_path = 'data/train'
X_train, Y_train = load_data(train_data_path)

# 构建模型
model = unet(input_size=(img_height, img_width, num_channels))

# 设置模型权重的保存路径
weights_path = 'unet_bubble_detector.keras'

# 如果先前保存的权重存在，则加载它们
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded model weights from:", weights_path)
else:
    print("No saved model weights found. Starting training from scratch.")

model.summary()  # 打印模型概要信息

# 设置模型保存点
model_checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1, save_best_only=True)

# 开始训练
model.fit(X_train, Y_train, batch_size=2, epochs=1, verbose=1, validation_split=0.1, callbacks=[model_checkpoint])
