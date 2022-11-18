import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from filters.quadratic_filter import quadratic_filter
from filters.TV_filter import TV_filter
from filters.non_local_means_filter import non_local_means_filter
from filters.non_local_wnnm_filter import non_local_wnnm_filter
from utilities.utils import read_image, add_noise, PSNR


def main(plot=False):
    np.random.seed(0)
    PSNR_results = {'quad':{}, 'TV':{}, 'nlm':{}, 'wnnm':{}}
    time_results = {'quad':{}, 'TV':{}, 'nlm':{}, 'wnnm':{}}

    for im_name in ['clock', 'boat', 'aerial', 'bridge', 'couple']:
        for key in PSNR_results.keys():
            PSNR_results[key][im_name] = []
            time_results[key][im_name] = []

        for var in [0.01, 0.025, 0.05]:
            im = read_image(im_name)
            noisy_im = add_noise(im, var)
            start_time = time.time()

            quad_im = quadratic_filter(noisy_im, 0.75)
            quad_time = time.time()
            TV_im = TV_filter(noisy_im, 0.5)
            TV_time = time.time()
            nlm_im = non_local_means_filter(noisy_im, 7, 10, 0.1)
            nlm_time = time.time()
            wnnm_im = non_local_wnnm_filter(noisy_im, 7, 10, 0.1)
            wnnm_time = time.time()

            PSNR_results['quad'][im_name].append(PSNR(original_im=im, cleaned_im=quad_im))
            PSNR_results['TV'][im_name].append(PSNR(original_im=im, cleaned_im=TV_im))
            PSNR_results['nlm'][im_name].append(PSNR(original_im=im, cleaned_im=nlm_im))
            PSNR_results['wnnm'][im_name].append(PSNR(original_im=im, cleaned_im=wnnm_im))

            time_results['quad'][im_name].append(quad_time - start_time)
            time_results['TV'][im_name].append(TV_time - quad_time)
            time_results['nlm'][im_name].append(nlm_time - TV_time)
            time_results['wnnm'][im_name].append(wnnm_time - nlm_time)

            if plot is True:
                _, ax_original = plt.subplots()
                ax_original.imshow(im, cmap='gray')
                ax_original.set_title('Original Image')

                _, ax_noisy = plt.subplots()
                ax_noisy.imshow(noisy_im, cmap='gray')
                ax_noisy.set_title('Noisy Image')

                _, ax_quad = plt.subplots()
                ax_quad.imshow(quad_im, cmap='gray')
                ax_quad.set_title('Quadratic Image')

                _, ax_tv = plt.subplots()
                ax_tv.imshow(TV_im, cmap='gray')
                ax_tv.set_title('TV Image')

                _, ax_nlm = plt.subplots()
                ax_nlm.imshow(nlm_im, cmap='gray')
                ax_nlm.set_title('Non-local means Image')

                _, ax_wnnm = plt.subplots()
                ax_wnnm.imshow(wnnm_im, cmap='gray')
                ax_wnnm.set_title('Weighted Nuclear Norm Minimization Image')

                plt.show()

    print(PSNR_results)
    print(time_results)
    with open('PSNR_results.pkl', 'wb') as f:
        pickle.dump(PSNR_results, f)
    with open('time_results.pkl', 'wb') as f:
        pickle.dump(time_results, f)


if __name__ == '__main__':
    main()
