import imageio
import numpy as np


def read_image(filename):
    im = imageio.imread(f'./images/{filename}.png')
    max_pixel = np.max(im)
    normalized_im = im/max_pixel
    return normalized_im

def add_noise(im, var=1.0):
    mean = 0
    sigma = var**0.5
    gaussian_noise = np.random.normal(mean, sigma, im.shape)
    noisy_im = im + gaussian_noise
    return noisy_im
