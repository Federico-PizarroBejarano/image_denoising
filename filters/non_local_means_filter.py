'''Non-Local Means Filter. '''
import numpy as np


def non_local_means_filter(im, patch_size, search_dist, h):
    '''Non-Local Means (NLM) filter.

    Args:
        im (np.ndarray): The noisy image to be filtered.
        patch_size (int): The size of patches to consider.
        search_dist (int): The distance from the center pixel of a patch to look.
        h (float): A constant used to calculate distance between patches.

    Returns:
        clean_im (np.ndarray): The filtered image.
    '''

    padding = patch_size // 2
    padded_im = np.pad(im, (padding, padding), mode='reflect')
    n, m = padded_im.shape

    clean_im = np.zeros(im.shape)

    for row in range(padding, n - padding):
        print(f'Completed {row}/{n} iterations.')
        for col in range(padding, m - padding):
            curr_patch = padded_im[row - padding:row + padding + 1, col - padding:col + padding + 1]
            total_sum = 0

            for s_row in range(row - search_dist, row + search_dist + 1):
                if s_row - padding < 0:
                    continue
                elif s_row + padding >= n:
                    break

                for s_col in range(col - search_dist, col + search_dist + 1):
                    if s_col - padding < 0 or (s_row == row and s_col == col):
                        continue
                    elif s_col + padding >= m:
                        break

                    search_patch = padded_im[s_row - padding:s_row + padding + 1, s_col - padding:s_col + padding + 1]

                    euclideanDistance = np.sqrt(np.sum(np.square(curr_patch - search_patch)))
                    weight = np.exp(-euclideanDistance / h)
                    total_sum += weight

                    clean_im[row - padding, col - padding] += weight * padded_im[s_row, s_col]

            clean_im[row - padding, col - padding] /= total_sum

    return clean_im
