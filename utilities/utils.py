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


def add_gaussian_noise(im, mean=0, var=0.01):
    '''Adds Gaussian noise to an image with the given variance.

    Args:
        mean (float): The mean of the Gaussian noise that is added.
        var (float): The variance of the Gaussian noise that is added.

    Returns:
        noisy_im (np.ndarray): The noisy image.
    '''
    sigma = var**0.5
    gaussian_noise = np.random.normal(mean, sigma, im.shape)
    noisy_im = im + gaussian_noise
    noisy_im = normalize_image(noisy_im)
    return noisy_im


def add_poisson_noise(im, photons=100):
    '''Adds Poisson noise to an image with the given variance.

    Args:
        photons (float): The number of photons available per pixel.

    Returns:
        noisy_im (np.ndarray): The noisy image.
    '''
    noisy_im = np.random.poisson(im * photons) / photons
    noisy_im = normalize_image(noisy_im)
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


def create_results_directory(noise_type, images, hyperparameters):
    '''Creates the necessary results directory structure.

    Args:
        noise_type (str): The type of noise, either 'gaussian' or 'poisson'.
        images (list): The list of image names, e.g. 'clock'.
        hyperparameters (list): The list of hyperparameters for that noise being tested.
    '''
    for image in images:
        for var in hyperparameters:
            str_var = str(var).replace('.', '_')
            os.makedirs(f'./results/{noise_type}/{image}/var_{str_var}/', exist_ok=True)
