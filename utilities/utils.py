'''A collection of utility functions. '''

import os
from math import log10, sqrt

import numpy as np
from PIL import Image


def read_image(filename):
    '''Reads one of the images, resizes them, and normalizes them to have pixels between 0 and 1.

    Args:
        filename (str): The name of the image to be read (e.g. clock).

    Returns:
        im (np.ndarray): The normalized, resized image.
    '''
    im = Image.open(f'./images/{filename}.png')
    im = im.resize((256, 256))
    im = np.array(im)
    im = normalize_image(im)
    return im


def add_noise(im, var=0.01):
    '''Adds Gaussian noise to an image with the given variance.

    Args:
        var (float): The variance of the Gaussian noise that is added.

    Returns:
        noisy_im (np.ndarray): The noisy image.
    '''
    mean = 0
    sigma = var**0.5
    gaussian_noise = np.random.normal(mean, sigma, im.shape)
    noisy_im = im + gaussian_noise
    noisy_im = np.clip(noisy_im, 0.0, 1.0)
    return noisy_im

def normalize_image(im):
    '''Normalizes an image to have pixels only from 0-1

    Args:
        im (np.ndarray): The image to be normalized.

    Returns:
        normalized_im (np.ndarray): The normalized image.
    '''
    max_pixel = np.max(im)
    min_pixel = np.min(im)
    normalized_im = (im - min_pixel) / max_pixel
    return normalized_im

def PSNR(original_im, cleaned_im):
    '''Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        original_im (np.ndarray): The original image (without any noise).
        cleaned_im (np.ndarray): The filtered image (trying to clear the noise).

    Returns:
        psnr (float): The normalized, resized image.
    '''
    mse = np.mean((original_im - cleaned_im)**2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def create_results_directory(images, variances):
    '''Creates the necessary results directory structure. '''
    for image in images:
        for var in variances:
            str_var = str(var).replace('.', '_')
            os.makedirs(f'./results/{image}/var_{str_var}/', exist_ok=True)
