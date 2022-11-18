'''Non-Local Weighted Nuclear Norm Minimization (WNNM) Filter. '''
import numpy as np


def non_local_wnnm_filter(im, patch_size, search_dist, h):
    '''Non-Local Weighted Nuclear Norm Minimization (WNNM) filter.

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
            similar_patches = _get_similar_patches(padded_im, patch_size, search_dist, row, col)
            clean_patches = _compute_wnnm(similar_patches, var=0.03)
            clean_patch = _collapse_stacked_patches(curr_patch, clean_patches, h)
            clean_im[row - padding, col - padding] = clean_patch[padding, padding]

    return clean_im


def _get_similar_patches(padded_im, patch_size, search_dist, row, col):
    '''Takes in a patch and returns all similar patches.

    Args:
        padded_im (np.ndarray): The padded image.
        patch_size (int): The size of patches to consider.
        search_dist (int): The distance from the center pixel of a patch to look.
        row (int): The row of the pixel being currently considered.
        col (int): The col of the pixel being currently considered.

    Returns:
        similar_patches (np.ndarray): The vertically stacked similar patches.
    '''

    padding = patch_size // 2
    n, m = padded_im.shape[0] - padding, padded_im.shape[1] - padding
    curr_patch = padded_im[row - padding:row + padding + 1, col - padding:col + padding + 1]

    all_patches = [(0, curr_patch)]

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
            all_patches.append((euclideanDistance, search_patch))

    all_patches.sort()
    similar_patches = np.vstack([item[1] for item in all_patches[0:10]])
    return similar_patches


def _compute_wnnm(stacked_patches, var):
    '''Given the stacked patches, calculates the minimum nuclear norm representation of the stack.

    Args:
        stacked_patches (np.ndarray): The vertically stacked similar patches.
        var (float): The variance of the Gaussian noise that has been added.

    Returns:
        clean_patches (np.ndarray): The vertically stacked 'clean' patches.
    '''

    u, sigmas_y, vh = np.linalg.svd(stacked_patches, full_matrices=True)
    sigmas_x = np.zeros(sigmas_y.shape[0])

    for i in range(sigmas_y.shape[0]):
        sigma_yi = sigmas_y[i]**0.5
        w_i = _estimate_weight(stacked_patches, var, sigma_yi)
        sigma_wi = max(sigma_yi**2 - w_i, 0)
        sigmas_x[i] = sigma_wi

    Sigma_X = np.diag(sigmas_x)
    Sigma_X = np.vstack((Sigma_X, np.zeros((u.shape[0] - Sigma_X.shape[0], Sigma_X.shape[1]))))
    clean_patches = u @ Sigma_X @ vh
    return clean_patches


def _estimate_weight(stacked_patches, var, sigma_yi):
    '''Calculates the weight necessary for the NNM process.

    Args:
        stacked_patches (np.ndarray): The vertically stacked similar patches.
        var (float): The variance of the Gaussian noise that has been added.
        sigma_yi (float): The ith singular value of stacked_patches.

    Returns:
        w_i (weight): The weight of the ith singular value of this WNNM.
    '''
    n = stacked_patches.shape[0] // stacked_patches.shape[1]
    c = 2**0.5
    eps = 0.0001
    sigma_xi = np.sqrt(max(sigma_yi**2 - n * var, 0))
    w_i = c * n**0.5 / (sigma_xi + eps)
    return w_i


def _collapse_stacked_patches(curr_patch, clean_patches, h):
    '''Using the cleaned patches, calculate the final clean patch.

    Args:
        curr_patch (np.ndarray): The current (noisy) patch being considered.
        clean_patches (np.ndarray): The vertically stacked clean similar patches.
        h (float): A constant used in calculating distance.

    Results:
        clean_patch (np.ndarray): The cleaned patch being considered.
    '''

    n = clean_patches.shape[0] // clean_patches.shape[1]
    total_sum = 0
    clean_patch = np.zeros(curr_patch.shape)

    for patch in range(n):
        search_patch = clean_patches[patch:patch + curr_patch.shape[0], :]
        euclideanDistance = np.sqrt(np.sum(np.square(curr_patch - search_patch)))
        weight = np.exp(-euclideanDistance / h)
        total_sum += weight
        clean_patch += weight * search_patch
    clean_patch /= total_sum
    return clean_patch
