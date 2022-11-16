import matplotlib.pyplot as plt

from filters.quadratic_filter import quadratic_filter
from filters.TV_filter import TV_filter
from utilities.utils import read_image, add_noise


def main():
    im = read_image('clock')
    noisy_im = add_noise(im, var=0.03)
    clean_im = quadratic_filter(noisy_im, 0.75)
    # clean_im = TV_filter(noisy_im, 0.5)

    _, ax_original = plt.subplots()
    ax_original.imshow(im, cmap='gray')

    _, ax_noisy = plt.subplots()
    ax_noisy.imshow(noisy_im, cmap='gray')

    _, ax_clean = plt.subplots()
    ax_clean.imshow(clean_im, cmap='gray')

    plt.show()  # display it


if __name__ == '__main__':
    main()
