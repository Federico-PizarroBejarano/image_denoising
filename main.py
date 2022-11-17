import numpy as np
import matplotlib.pyplot as plt

from filters.quadratic_filter import quadratic_filter
from filters.TV_filter import TV_filter
from filters.non_local_means_filter import non_local_means_filter
from filters.non_local_wnnm_filter import non_local_wnnm_filter
from utilities.utils import read_image, add_noise


def main():
    im = read_image('clock')
    np.random.seed(0)
    noisy_im = add_noise(im, var=0.03)
    # clean_im = quadratic_filter(noisy_im, 0.75)
    # clean_im = TV_filter(noisy_im, 0.5)
    # clean_im = non_local_means_filter(noisy_im, 7, 10, 0.1)
    clean_im = non_local_wnnm_filter(noisy_im, 7, 10, 0.1)

    _, ax_original = plt.subplots()
    ax_original.imshow(im, cmap='gray')

    _, ax_noisy = plt.subplots()
    ax_noisy.imshow(noisy_im, cmap='gray')

    _, ax_clean = plt.subplots()
    ax_clean.imshow(clean_im, cmap='gray')

    plt.show()  # display it


if __name__ == '__main__':
    main()
