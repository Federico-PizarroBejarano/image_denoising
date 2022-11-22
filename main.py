'''Main file for running experiments.'''

import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

from filters.quadratic_filter import quadratic_filter
from filters.TV_filter import TV_filter
from filters.TV_filter_pd import TV_filter_pd
from filters.non_local_means_filter import non_local_means_filter
from filters.non_local_wnnm_filter import non_local_wnnm_filter
from utilities.utils import read_image, add_noise, PSNR, create_results_directory, normalize_image


def main(plot=False, savefigs=True):
    '''Runs the various filters on the provided images with varying noise levels
       and saves the results.

    Args:
        plot (bool): Whether to plot the noisy and cleaned images.
        savefigs (bool): Whether to save the generated images.
    '''

    np.random.seed(0)

    PSNR_results = {'quad': {}, 'TV': {}, 'nlm': {}, 'wnnm': {}}
    time_results = {'quad': {}, 'TV': {}, 'nlm': {}, 'wnnm': {}}

    images = ['clock', 'boat', 'aerial', 'bridge', 'couple']
    variances = [0.01, 0.025, 0.05]

    create_results_directory(images, variances)

    for im_name in images:
        for key in PSNR_results.keys():
            PSNR_results[key][im_name] = []
            time_results[key][im_name] = []

        for var in variances:
            str_var = str(var).replace('.', '_')

            im = read_image(im_name)
            noisy_im = add_noise(im, var)
            start_time = time.time()

            quad_im = quadratic_filter(noisy_im, 0.75)
            quad_time = time.time()
            quad_im = normalize_image(quad_im)

            # TV_im = TV_filter(noisy_im, 1.5)
            TV_im = TV_filter_pd(noisy_im, 6)
            TV_time = time.time()
            TV_im = normalize_image(TV_im)

            nlm_im = non_local_means_filter(noisy_im, 7, 10, 0.1)
            nlm_time = time.time()
            nlm_im = normalize_image(nlm_im)

            x = noisy_im
            y = noisy_im
            delta = 0.3
            for _ in range(1):
                y = x + delta*(noisy_im - y)
                x = non_local_wnnm_filter(y, 7, 10, var)
                x = normalize_image(x)

            wnnm_im = x

            wnnm_time = time.time()

            PSNR_results['quad'][im_name].append(PSNR(original_im=im, cleaned_im=quad_im))
            PSNR_results['TV'][im_name].append(PSNR(original_im=im, cleaned_im=TV_im))
            PSNR_results['nlm'][im_name].append(PSNR(original_im=im, cleaned_im=nlm_im))
            PSNR_results['wnnm'][im_name].append(PSNR(original_im=im, cleaned_im=wnnm_im))

            time_results['quad'][im_name].append(quad_time - start_time)
            time_results['TV'][im_name].append(TV_time - quad_time)
            time_results['nlm'][im_name].append(nlm_time - TV_time)
            time_results['wnnm'][im_name].append(wnnm_time - nlm_time)

            if plot is True or savefigs is True:
                _, ax_original = plt.subplots()
                ax_original.imshow(im, cmap='gray')
                ax_original.set_title('Original Image')

                fig_noisy, ax_noisy = plt.subplots()
                ax_noisy.imshow(noisy_im, cmap='gray')
                ax_noisy.set_title(f'Noisy Image, PSNR={round(PSNR(original_im=im, cleaned_im=noisy_im), 2)}')

                fig_quad, ax_quad = plt.subplots()
                ax_quad.imshow(quad_im, cmap='gray')
                ax_quad.set_title(f'Quadratic Image, PSNR={round(PSNR(original_im=im, cleaned_im=quad_im), 2)}')

                fig_tv, ax_tv = plt.subplots()
                ax_tv.imshow(TV_im, cmap='gray')
                ax_tv.set_title(f'TV Image, PSNR={round(PSNR(original_im=im, cleaned_im=TV_im), 2)}')

                fig_nlm, ax_nlm = plt.subplots()
                ax_nlm.imshow(nlm_im, cmap='gray')
                ax_nlm.set_title(f'Non-local means Image, PSNR={round(PSNR(original_im=im, cleaned_im=nlm_im), 2)}')

                fig_wnnm, ax_wnnm = plt.subplots()
                ax_wnnm.imshow(wnnm_im, cmap='gray')
                ax_wnnm.set_title(f'Weighted Nuclear Norm Minimization Image, PSNR={round(PSNR(original_im=im, cleaned_im=wnnm_im), 2)}')

                if savefigs is True:
                    fig_noisy.savefig(f'./results/{im_name}/var_{str_var}/noisy.png')
                    fig_quad.savefig(f'./results/{im_name}/var_{str_var}/quad.png')
                    fig_tv.savefig(f'./results/{im_name}/var_{str_var}/tv.png')
                    fig_nlm.savefig(f'./results/{im_name}/var_{str_var}/nlm.png')
                    fig_wnnm.savefig(f'./results/{im_name}/var_{str_var}/wnnm.png')

                if plot is True:
                    plt.show()

    print(PSNR_results)
    print(time_results)
    with open('./results/PSNR_results.pkl', 'wb') as f:
        pickle.dump(PSNR_results, f)
    with open('./results/time_results.pkl', 'wb') as f:
        pickle.dump(time_results, f)


if __name__ == '__main__':
    main()
