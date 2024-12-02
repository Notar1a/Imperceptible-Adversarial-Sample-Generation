import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_mse(img1, img2):
    """计算均方误差（MSE）"""
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=2, win_size=3)

def evaluate_image_quality(original_img_path, perturbed_img_path):
    # 读取图片
    original_img = cv2.imread(original_img_path)
    perturbed_img = cv2.imread(perturbed_img_path)

    # 转换为灰度图像以计算PSNR和MSE
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    perturbed_gray = cv2.cvtColor(perturbed_img, cv2.COLOR_BGR2GRAY)

    # 计算MSE、PSNR和SSIM
    mse_value = calculate_mse(original_gray, perturbed_gray)
    psnr_value = calculate_psnr(original_gray, perturbed_gray)
    ssim_value = calculate_ssim(original_img, perturbed_img)

    # 打印结果
    print(f"MSE: {mse_value:.4f}")
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_value:.4f}")

    return mse_value, psnr_value, ssim_value

# 示例路径
original_img_path = 'data/cropped-CUTE80/163_years_years.png'
perturbed_img_path = '20241129_163932_STARNet-TPS-ResNet-BiLSTM-CTC-sensitive.pth/downsampled_images/163_years_years.png'

# 评估图片质量
mse_value, psnr_value, ssim_value = evaluate_image_quality(original_img_path, perturbed_img_path)
