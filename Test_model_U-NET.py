import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Kích thước ảnh
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Load model đã train
model = tf.keras.models.load_model("unet_brain_mri.h5")

# Load ảnh MRI để test
def load_test_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # Thêm batch dimension

# Đường dẫn ảnh test
TEST_IMAGE_PATH = "dataset/test/TCGA_CS_4941_19960909_12.tif"

# Load ảnh test
test_image = load_test_image(TEST_IMAGE_PATH)

# Dự đoán mask
predicted_mask = model.predict(test_image)[0]

# Hiển thị ảnh gốc và mask dự đoán
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(test_image[0, :, :, 0], cmap="gray")
plt.title("Ảnh MRI gốc")

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask[:, :, 0], cmap="gray")
plt.title("Mask dự đoán")

plt.show()
