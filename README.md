# Image Denoising
Testing different image denoising techniques through the lens of convex optimization.

## Techniques
Consider a greyscale image Im. We are looking for a denoised image X. We have implemented the following techniques:
- **Quadratic Filtering:** Solving the optimization min<sub>X</sub> ||X - Im|| + &lambda; ||&nabla;X||<sup>2</sup>
- **Total Variation (TV) Filtering:** Solving the optimization min<sub>X</sub> ||X - Im|| + &lambda; ||&nabla;X||
- **Non-Local Means (NLM) Filtering:** Finding similar patches to every patch in the image and using a mean filter to smooth them.
- **Non-Local Weighted Nuclear Norm Minimization WNNM):** Finding similar patches to every patch in the image and using a weighted nuclear norm minimization to remove noisy singular values using a quadratic program.

## Solving Techniques
Quadratic filtering and TV filtering are bicriteria optimizations since they have two different objectives. To solve this, we have implemented the primal-dual algorithm and the alternating direction method of multipliers (ADMM), and compared these approaches with the out-of-the-box convex optimization.

## Testing
We compare the various approaches on the following images, with additive white Gaussian noise (zero-mean and varying but known variance). We compare the efficacy of the approaches both qualitatively, and using the peak signal to noise ratio (PSNR). The results are:

Our final report can be found here:
