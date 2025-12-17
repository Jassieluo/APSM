import cv2
import os
from skimage import util
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import io
from tqdm import tqdm
from typing import List, Dict, Union


# Define folder path
folder_path = '/home/lqs/Datasets/ShipRSImageNet-YOLO/images/val/'
# Define output folder path for noisy images
noise_folder_path= '/home/lqs/Datasets/ShipRSImageNet-YOLO-NOISE/images/val/'

def add_salt_and_pepper_noise(image, noise_ratio):
    row, col, channel = image.shape
    s_vs_p = 0.5
    amount = noise_ratio
    out = np.copy(image)
    # Salt noise
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[coords[0], coords[1], :] = 1

    # Pepper noise
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[coords[0], coords[1], :] = 0
    return out


def add_multiple_noises(image: np.ndarray, noise_sequence: List[Dict[str, Union[str, float]]]) -> np.ndarray:
    # Ensure image is float type in range [0, 1]
    if image.dtype == np.uint8:
        noisy_img = image.astype(np.float32) / 255.0
    else:
        noisy_img = image.copy()

    # Add multiple noise types sequentially
    for noise_config in noise_sequence:
        noise_type = noise_config['type']
        params = noise_config.get('params', {})

        if noise_type == 'gaussian':
            mean = params.get('mean', 0)
            var = params.get('var', 0.01)
            noisy_img = util.random_noise(noisy_img, mode='gaussian', mean=mean, var=var)

        elif noise_type == 'salt':
            amount = params.get('amount', 0.05)
            noisy_img = util.random_noise(noisy_img, mode='salt', amount=amount)

        elif noise_type == 'pepper':
            amount = params.get('amount', 0.05)
            noisy_img = util.random_noise(noisy_img, mode='pepper', amount=amount)

        elif noise_type == 's&p':
            amount = params.get('amount', 0.05)
            noisy_img = util.random_noise(noisy_img, mode='s&p', amount=amount)

        elif noise_type == 'speckle':
            noisy_img = util.random_noise(noisy_img, mode='speckle')

        else:
            raise ValueError(f"未知噪声类型: {noise_type}")

    # Convert back to OpenCV format (0-255, uint8)
    noisy_img = (noisy_img * 255).clip(0, 255).astype(np.uint8)
    return noisy_img



# Process all images in the folder
for filename in tqdm(os.listdir(folder_path)):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp'):  # 只处理jpg和png格式的图像
        img_path = os.path.join(folder_path, filename)

        img_ori = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        noise_image = add_multiple_noises(image=gray_img, noise_sequence=[
            {"type": "gaussian", "params": {"var": 0.002}},
            {"type": "pepper", "params": {"amount": 0.002}},
            {"type": "speckle", "params": {"amount": 0.002}},
        ])
        # Save noisy image
        cv2.imwrite(os.path.join(noise_folder_path, filename), noise_image)


