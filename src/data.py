"""
数据生成模块（v3.0）：构造合成图像去模糊数据集。
与 v2.0 兼容，保留 Cameraman + Gaussian 模糊 + 脉冲噪声设置。
"""

import numpy as np
from skimage import data
from skimage.transform import resize


def get_cameraman(size=(256, 256)):
    img = data.camera()
    img = img.astype(np.float64) / 255.0
    if img.shape != size:
        img = resize(img, size, mode='reflect', anti_aliasing=True)
    return img


def gaussian_kernel(size=15, sigma=1.5):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def add_salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    noisy = np.copy(image)
    num_salt = int(np.ceil(amount * image.size * salt_vs_pepper))
    num_pepper = int(np.ceil(amount * image.size * (1.0 - salt_vs_pepper)))
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 1.0
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0.0
    return noisy


def generate_deblurring_data(size=(256, 256), blur_sigma=1.5, blur_size=15,
                             noise_amount=0.07, seed=42):
    np.random.seed(seed)
    x_true = get_cameraman(size)
    kernel = gaussian_kernel(blur_size, blur_sigma)
    x_fft = np.fft.fft2(x_true)
    k_fft = np.fft.fft2(kernel, s=size)
    b_clean = np.real(np.fft.ifft2(x_fft * k_fft))
    b_clean = np.clip(b_clean, 0.0, 1.0)
    b = add_salt_pepper_noise(b_clean, amount=noise_amount)
    return x_true, kernel, b
