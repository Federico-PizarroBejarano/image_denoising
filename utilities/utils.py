from PIL import Image
import numpy as np
from math import log10, sqrt


def read_image(filename):
    im = Image.open(f'./images/{filename}.png')
    im = im.resize((256, 256))
    im = np.array(im)
    max_pixel = 255.0
    normalized_im = im/max_pixel
    return normalized_im

def add_noise(im, var=1.0):
    mean = 0
    sigma = var**0.5
    gaussian_noise = np.random.normal(mean, sigma, im.shape)
    noisy_im = im + gaussian_noise
    noisy_im = np.clip(noisy_im, 0.0, 1.0)
    return noisy_im

def PSNR(original_im, cleaned_im):
    mse = np.mean((original_im - cleaned_im)**2)
    if(mse == 0):
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
