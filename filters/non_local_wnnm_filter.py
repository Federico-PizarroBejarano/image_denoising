import numpy as np


def get_similar_patches(padded_im, patch_size, search_dist, row, col, tol):
    padding = patch_size//2
    n, m = padded_im.shape[0] - padding, padded_im.shape[1] - padding
    curr_patch = padded_im[row-padding:row+padding+1, col-padding:col+padding+1]

    similar_patches = curr_patch

    for s_row in range(row-search_dist, row+search_dist+1):
        if s_row-padding < 0:
            continue
        elif s_row+padding >= n:
            break

        for s_col in range(col-search_dist, col+search_dist+1):
            if s_col-padding < 0 or (s_row==row and s_col==col):
                continue
            elif s_col+padding >= m:
                break

            search_patch = padded_im[s_row-padding:s_row+padding+1, s_col-padding:s_col+padding+1]
            euclideanDistance = np.sqrt(np.sum(np.square(curr_patch - search_patch)))

            if euclideanDistance <= tol:
                similar_patches = np.vstack((similar_patches, search_patch))

    return similar_patches


def estimate_weight(stacked_patches, i, var, sigma_yi):
    n = stacked_patches.shape[0]//stacked_patches.shape[1]
    c = 2**0.5
    eps = 0.0001
    sigma_xi = np.sqrt(max(sigma_yi**2 - n*var, 0))
    w_i = c*n**0.5/(sigma_xi + eps)
    return w_i


def compute_wnnm(stacked_patches, var):
    u, sigmas_y, vh = np.linalg.svd(stacked_patches, full_matrices=True)
    sigmas_x = np.zeros(sigmas_y.shape[0])

    for i in range(sigmas_y.shape[0]):
        sigma_yi = sigmas_y[i]**0.5
        w_i = estimate_weight(stacked_patches, i, var, sigma_yi)
        sigma_wi = max(sigma_yi**2 - w_i, 0)
        sigmas_x[i] = sigma_wi

    Sigma_X = np.diag(sigmas_x)
    Sigma_X = np.vstack((Sigma_X, np.zeros((u.shape[0] - Sigma_X.shape[0], Sigma_X.shape[1]))))
    clean_patches = u @ Sigma_X @ vh
    return clean_patches


def collapse_stacked_patches(curr_patch, clean_patches, h):
    n = clean_patches.shape[0]//clean_patches.shape[1]
    total_sum = 0
    clean_patch = np.zeros(curr_patch.shape)

    for patch in range(n):
        search_patch = clean_patches[patch:patch+curr_patch.shape[0], :]
        euclideanDistance = np.sqrt(np.sum(np.square(curr_patch - search_patch)))
        weight = np.exp(-euclideanDistance/h)
        total_sum += weight
        clean_patch += weight*search_patch
    clean_patch /= total_sum
    return clean_patch


def non_local_wnnm_filter(im, patch_size, search_dist, h):
    padding = patch_size//2
    padded_im = np.pad(im, (padding, padding), mode="reflect")
    n, m = padded_im.shape

    clean_im=np.zeros(im.shape)

    for row in range(padding, n-padding):
        print(f'Completed {row}/{n} iterations.')
        for col in range(padding, m-padding):
            curr_patch = padded_im[row-padding:row+padding+1, col-padding:col+padding+1]
            similar_patches = get_similar_patches(padded_im, patch_size, search_dist, row, col, tol=1.5)
            clean_patches = compute_wnnm(similar_patches, var=0.03)
            clean_patch = collapse_stacked_patches(curr_patch, clean_patches, h)
            clean_im[row-padding, col-padding] = clean_patch[padding, padding]

    return clean_im
