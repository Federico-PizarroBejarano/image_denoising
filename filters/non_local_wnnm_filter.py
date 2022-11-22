'''Non-Local Weighted Nuclear Norm Minimization (WNNM) Filter. '''
import numpy as np


def non_local_wnnm_filter(im, patch_size, search_dist, var):
    '''Non-Local Weighted Nuclear Norm Minimization (WNNM) filter.

    Args:
        im (np.ndarray): The noisy image to be filtered.
        patch_size (int): The size of patches to consider.
        search_dist (int): The distance from the center pixel of a patch to look.
        var (float): The variance of the noise on the image.

    Returns:
        clean_im (np.ndarray): The filtered image.
    '''

    pad = patch_size // 2
    padded_im = np.pad(im, (pad, pad), mode='reflect')
    n, m = padded_im.shape

    clean_padded_im = np.zeros(padded_im.shape)
    count_padded_im = np.zeros(padded_im.shape)

    for row in range(pad, n - pad):
        print(f'Completed {row}/{n} iterations.')
        for col in range(pad, m - pad):
            curr_patch = padded_im[row - pad:row + pad + 1, col - pad:col + pad + 1]
            similar_patches, patch_locations = _get_similar_patches(padded_im, patch_size, search_dist, row, col)
            clean_patches = _compute_wnnm(similar_patches, var)
            im_update, count_update = _collapse_stacked_patches(padded_im.shape, curr_patch, clean_patches, patch_locations)
            clean_padded_im += im_update
            count_padded_im += count_update

    clean_padded_im /= count_padded_im
    clean_im = clean_padded_im[pad:-pad, pad:-pad]

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
        patch_locations (list): The location of the patches.
    '''

    pad = patch_size // 2
    n, m = padded_im.shape[0] - pad, padded_im.shape[1] - pad
    curr_patch = padded_im[row - pad:row + pad + 1, col - pad:col + pad + 1]

    all_patches = [(0, (row, col), curr_patch)]

    for s_row in range(row - search_dist, row + search_dist + 1):
        if s_row - pad < 0:
            continue
        elif s_row + pad >= n:
            break

        for s_col in range(col - search_dist, col + search_dist + 1):
            if s_col - pad < 0 or (s_row == row and s_col == col):
                continue
            elif s_col + pad >= m:
                break

            search_patch = padded_im[s_row - pad:s_row + pad + 1, s_col - pad:s_col + pad + 1]
            euclideanDistance = np.sqrt(np.sum(np.square(curr_patch - search_patch)))
            if euclideanDistance <= 1.75:
                all_patches.append((euclideanDistance, (s_row, s_col), search_patch))

    all_patches.sort()
    similar_patches = np.vstack([item[2] for item in all_patches[0:min(len(all_patches), 10)]])
    patch_locations = [item[1] for item in all_patches[0:min(len(all_patches), 10)]]
    return similar_patches, patch_locations


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
        sigma_yi = sigmas_y[i]
        w_i = _estimate_weight(stacked_patches, var, sigma_yi)
        sigma_wi = max(sigma_yi - w_i, 0)
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
    eps = 0.00001
    sigma_xi = np.sqrt(max(sigma_yi**2 - n * var, 0))
    w_i = (2 * n)**0.5 / (sigma_xi + eps)
    return w_i


def _collapse_stacked_patches(padded_im_shape, curr_patch, clean_patches, patch_locations):
    '''Using the cleaned patches, calculate the final clean patch.

    Args:
        padded_im_shape (tuple): The shape of the padded image.
        curr_patch (np.ndarray): The current (noisy) patch being considered.
        clean_patches (np.ndarray): The vertically stacked clean similar patches.
        patch_locations (list): The location of the patches.

    Results:
        clean_patch (np.ndarray): The cleaned patch being considered.
    '''

    patch_size = clean_patches.shape[1]
    n = clean_patches.shape[0] // patch_size
    pad = clean_patches.shape[1] // 2
    updated_im = np.zeros(padded_im_shape)
    updated_count = np.zeros(padded_im_shape)

    for patch in range(n):
        clean_patch = clean_patches[patch:patch + patch_size, :]
        euclideanDistance = np.sqrt(np.sum(np.square(curr_patch - clean_patch)))
        weight = np.exp(-euclideanDistance / 0.1)
        s_row, s_col = patch_locations[patch]
        updated_im[s_row - pad:s_row + pad + 1, s_col - pad:s_col + pad + 1] += weight * clean_patch
        updated_count[s_row - pad:s_row + pad + 1, s_col - pad:s_col + pad + 1] += weight * np.ones((patch_size, patch_size))

    return updated_im, updated_count
