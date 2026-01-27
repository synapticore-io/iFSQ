import numpy as np
import torch
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim_loss

def calculate_ssim(img1, img2, channel_axis=-1):
    ssim_temp = 0
    B, _, _, _ = img1.shape
    for i in range(B):
        rgb_restored_s, rgb_gt_s = img1[i], img2[i]
        ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s,data_range=1.0, channel_axis=channel_axis)
    return ssim_temp / B
 
 