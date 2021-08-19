# `absorption_image_processor`
Python implementation of absorption image processing via PCA with properly normalized atom region masking.

Much of the programming structure in this code, such as the API and memory management, is inspired by [Chris Billington's image_reconstruction repo](https://github.com/chrisjbillington/image_reconstruction).
However, the data analysis algorithm is very different.
Instead it uses the same approach as in the original [MATLAB version](https://github.com/zakv/AbsorptionImageProcessing_MATLAB) of this code.
The MATLAB repo's README and the docstrings of its functions include a brief explanation of the math behind the approach.
A full explanation is provided in Appendix D of my thesis ("Raman Cooling to High Phase Space Density" by Zachary Vendeiro), available online.
See the module-level docstring of the `absorption_image_processor.py` file in this repo for an example showing how to use the code in this Python repo.

The advantage of the code in this repo compared to more typical absorption imaging PCA analysis routines<sup>[1](#footnote1)</sup> is that it properly handles masking the region of the image with atoms.
In particular it does so in a way that does not introduce the systematic effects which occur when simply zeroing all of the pixels which may have atoms during the PCA reconstruction.
See the documentation in the [MATLAB implementation repo](https://github.com/zakv/AbsorptionImageProcessing_MATLAB) for more information.


<a name="footnote1">1</a>: Note that [Chris Billington's image_reconstruction](https://github.com/chrisjbillington/image_reconstruction) does *not* suffer from this normalization issue and may work as well or better than the code in this repo.
