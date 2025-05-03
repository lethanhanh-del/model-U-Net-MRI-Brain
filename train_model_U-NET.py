import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Kích thước ảnh đầu vào
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Đường dẫn thư mục chứa ảnh
IMG_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
IMG_SIZE = (128, 128)  # Resize về 128x128

# Load ảnh và mask
def load_dataset(img_dir, mask_dir):
    img_files = sorted(os.listdir(img_dir))  # Sắp xếp file để đảm bảo trùng khớp
    mask_files = sorted(os.listdir(mask_dir))

    X, Y = [], []
    for img_file, mask_file in zip(img_files, mask_files):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Đọc ảnh và mask
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize và chuẩn hóa về [0, 1]
        img = cv2.resize(img, IMG_SIZE) / 255.0
        mask = cv2.resize(mask, IMG_SIZE) / 255.0

        # Thêm kênh màu thứ 3 (vì U-Net yêu cầu input có 3 chiều)
        X.append(np.expand_dims(img, axis=-1))
        Y.append(np.expand_dims(mask, axis=-1))

    return np.array(X), np.array(Y)

# Load dataset
X, Y = load_dataset(IMG_DIR, MASK_DIR)

# Tạo mô hình U-Net đơn giản
def build_unet():
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, 1))

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c2 = layers.MaxPooling2D((2, 2))(c2)

    # Decoder
    c3 = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(c2)
    c4 = layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), activation='sigmoid', padding='same')(c3)

    model = models.Model(inputs, c4)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# Load dữ liệu
X, Y = load_dataset(IMG_DIR, MASK_DIR)

# Chia thành tập train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Khởi tạo và train model
model = build_unet()
model.fit(X_train, Y_train, epochs=10, batch_size=8, validation_data=(X_test, Y_test))

# Lưu model sau khi train
model.save("unet_brain_mri.h5")
print("✅ Model đã được train và lưu thành công!")
