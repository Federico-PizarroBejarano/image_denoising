'''Total Variation (L1) Filter'''
import numpy as np


def TV_filter_pd(im, lamb=1, niter=100):
    '''Total Variation (L1) Filter coded using the primal dual algorithm.

    Args:
        im (np.ndarray): The noisy image to be filtered.
        lamb (float): The free parameter (lambda) that determines how much to correct.
        niter (int): The number of iterations to run.

    Returns:
        clean_im (np.ndarray): The filtered image.
    '''
    tau = 0.02
    sigma = 1.0 / (8.0 * tau)
    lt = lamb * tau
    n = im.shape[0]

    clean_im = im
    shift = list(range(1, n)) + [n - 1]

    p = np.zeros((n, n, 2))
    p[:, :, 0] = clean_im[:, shift] - clean_im
    p[:, :, 1] = clean_im[shift, :] - clean_im

    for _ in range(niter):
        ux = clean_im[:, shift] - clean_im
        uy = clean_im[shift, :] - clean_im
        p = p + sigma * np.dstack([ux, uy])

        normep = np.maximum(1, np.sqrt(np.square(p[:, :, 0]) + np.square(p[:, :, 1])))
        p[:, :, 0] = p[:, :, 0] / normep
        p[:, :, 1] = p[:, :, 1] / normep

        div = np.vstack((p[0:n - 1, :, 1], np.zeros((1, n)))) - np.vstack((np.zeros((1, n)), p[0:n - 1, :, 1]))
        div = np.hstack((p[:, 0:n - 1, 0], np.zeros((n, 1)))) - np.hstack((np.zeros((n, 1)), p[:, 0:n - 1, 0])) + div

        clean_im = (clean_im + tau * div + lt * im) / (1 + tau)
    return clean_im
