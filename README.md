# Image Denoising
Testing different image denoising techniques through the lens of convex optimization.

## Installation

### Clone repo
```bash
git clone https://github.com/Federico-PizarroBejarano/image_denoising.git
cd image_denoising
```

### Create a Conda Environment
Create and access a Python 3.8 environment using
[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

```bash
conda create -n denoise python=3.8.10
conda activate denoise
```

Install the `image_denoising` repository

```
pip install --upgrade pip
pip install -r requirements.txt
```

### Install Mosek
Obtain [MOSEK's license](https://www.mosek.com/products/academic-licenses/) (free for academia).
Once you have received (via e-mail) and downloaded the license to your own `~/Downloads` folder, install it by executing
```
$ mkdir ~/mosek                                                    # Create MOSEK license folder in your home '~'
$ mv ~/Downloads/mosek.lic ~/mosek/                                # Copy the downloaded MOSEK license to '~/mosek/'
```

## Denoising Techniques
Consider a greyscale image Y. We are looking for a denoised image X. We have implemented the following techniques:
- **Quadratic Filtering:** Solve the optimization - min<sub>X</sub> ||X - Y||<sub>F</sub> + &lambda; ||&nabla;X||<sub>2</sub>

- **Total Variation (TV) Filtering:** Solve the optimization - min<sub>X</sub> ||X - Y||<sub>1</sub> + &lambda; ||&nabla;X||<sub>1</sub>

- **Non-Local Means (NLM) Filtering:** Finding similar patches to every patch in the image and using a mean filter to smooth them.
- **Non-Local Weighted Nuclear Norm Minimization WNNM):** Finding similar patches to every patch in the image and using a weighted nuclear norm minimization to remove noisy singular values using a quadratic program.

## Optimization Techniques
Quadratic filtering and TV filtering are bicriteria optimizations since they have two different objectives. To solve this, we have implemented the primal-dual algorithm and the alternating direction method of multipliers (ADMM), and compared these approaches with the out-of-the-box convex optimization.

## Testing
We compare the various approaches on the following images, with additive white Gaussian noise (zero-mean and varying but known variance). We compare the efficacy of the approaches both qualitatively, and using the peak signal to noise ratio (PSNR). The results are:

Our final report can be found here:
