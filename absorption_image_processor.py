r"""Module for using principal component analysis to process absorption images.

Much of this module's code is inspired by Chris Billington's
"image_reconstruction" repository on his github, which is available at
https://github.com/chrisjbillington/image_reconstruction as of this writing. The
overall structure, API, and memory management is based on that code, but the
actual implementation of the data analysis has been changed significantly. The
new implementation follows the same approach as the MATLAB code from
https://github.com/zakv/AbsorptionImageProcessing_MATLAB. For more information
see that project's repo. In particular the README in that repo explains the math
behind the approach, and additional information can be found in the docstrings
of the functions there. A special feature of this PCA implementation is that it
properly handles normalization issues that can occur when masking the atom
region, as explained in the README of the MATLAB version repo.

Example Usage:
```
# Be sure to run Python from the directory containing this file, or add it to
# the Python path, so that it can be imported.
from absorption_image_processor import AbsorptionImageProcessor

# Other imports.
import matplotlib.pyplot as plt

# Define a function for plotting images.
def plot_array(array):
    plt.figure()
    plt.imshow(array)
    plt.colorbar()

# Create an instance of the AbsorptionImageProcessor class.
max_beam_images = 1000  # Adjust as desired.
max_principal_components = 100  # Adjust as desired.
image_processor = AbsorptionImageProcessor(
    max_beam_images=max_beam_images,
    max_principal_components=max_principal_components,
    use_sparse_routines=True,
)
# It may be worth testing whether setting use_sparse_routines to True or False
# makes the analysis run faster for your system.

# Fill up the processor with beam image data. It is assumed that beam_image_list
# is a list of 2D images of the absorption imaging beam without atoms. The user
# must generate this list from their data. If an image is taken with the beam
# extinguished to measure any background signal, that should be subtracted from
# the beam and atom images before passing them to the AbsorptionImageProcessor.
image_processor.add_beam_images(beam_image_list)
# Alternatively beam images can be added individually, as shown below.
# for beam_image in beam_image_list:
#     image_processor.add_beam_image(beam_image)

# Set the atom region mask.
atom_region_rows = [9, 120]  # Adjust as desired.
atom_region_cols = [49, 280]  # Adjust as desired.
image_processor.set_rectangular_mask(atom_region_rows, atom_region_cols)

# Perform the PCA to get the mean image and eigenvectors.
mean_image, mask_used, principal_component_images, variances = image_processor.get_pca_images()

# Display some results.
plot_array(mask_used)
plt.title("Mask")
plot_array(mean_image)
plt.title("Mean Image")
plot_array(principal_component_images[0])
plt.title("Principal Component #0")
plot_array(principal_component_images[1])
plt.title("Principal Component #1")
plot_array(principal_component_images[-1])
plt.title("Smallest Principal Component")

# Plot that variances associated with each of the principal components.
variances = image_processor.get_pca_variances()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(variances)
ax.set_yscale('log')
ax.set_title("Contributions of Principal Components")
ax.set_xlabel("Principal Component Index")
ax.set_ylabel("Variance of Principal Component")

# As a test, try to reconstruct one of the beam images as if it were an atom
# image. Ideally the reconstruction should look identical to the original (up to
# differences due to noise) because there are no atoms in the image.
n_images = len(beam_image_list)
test_image_index = int(n_images/2)
test_image = beam_image_list[test_image_index]

# Do the reconstruction.
reconstruction = image_processor.reconstruct(test_image)

# Plot the test image that was reconstructed.
plot_array(test_image)
plt.title("Test Image")

# Plot the reconstruction.
plot_array(reconstruction)
plt.title("Reconstructed Image")

# Plot the difference with the original, which should be zero except where there
# are atoms. Given that there are no atoms in the test image, it should then be
# zero everywhere, except for noise.
plot_array(reconstruction - test_image)
plt.title("Residual After Reconstruction")

# The interesting quantity though is not the number of missing photons, but the
# optical depth, so plot that. For the test image without atoms, this should be
# zero everywhere, except for noise.
od_image = image_processor.get_od_image(test_image)
plot_array(od_image)
plt.title("Optical Depth")
```
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange
from scipy.linalg import eigh
from scipy.sparse.linalg.eigen.arpack import eigsh

# TODO:
# * Speed optimization:
#     * Use a profiler to find the time-consuming sections.
#     * Compare the dense and sparse eigenvector algorithms.
#     * Plot with numba more to see if it can be faster.
# * Figure out how to do PCA with mask in pixel basis instead of image basis:
#     * Maybe can diagonalize in pixel basis, then convert to image basis by
#       left-multiplying by A.T? Then can use same masking trick to get back to
#       pixel basis?
#     * Should take this approach when the number of images is larger than the
#       number of pixels in each image.


class AbsorptionImageProcessor():
    """A class for performing analysis of data from absorption imaging.

    This class is built to manage the data from many runs of an experiment which
    uses absorption imaging to image ultracold atomic clouds. For additional
    information, see the docstring of the module that contains this class.
    """

    def __init__(self, max_beam_images=1000, max_principal_components=100,
                 use_sparse_routines=True):
        """Create and AbsorptionImageProcessor instance.

        Args:
            max_beam_images (int, optional): (Default = 1000) The maximum number
                of beam images to use as inputs to the PCA. Note that this also
                sets the size of a preallocated array used to store the data
                from these images. This array can take up a lot of memory if
                max_beam_images is set to a large value.
            max_principal_components (int, optional): (Default = 100) The
                maximum number of principal components to use when
                reconstructing images. Fewer may be used if there are not enough
                beam images to create that many principal components
            use_sparse_routines (bool, optional): (Default = True) Sets whether
                the scipy routines for dense or sparse matrices are used to
                compute the principal components. Surprisingly the sparse
                routine can be faster than the dense one even for dense
                matrices. That may be more likely when the number of principal
                components calculated is much smaller than the number of beam
                images.
        """
        self._max_beam_images = max_beam_images
        self._max_principal_components = max_principal_components
        self._use_sparse_routines = use_sparse_routines
        self._initialised = False
        self._mark_cache_invalid()

    def _init(self, beam_image):
        """Initialize attributes that depend on image dimensions

        This method is called when the first beam image is added to the given
        instance. Doing this outside of the real __init__() method means that
        an instance of this class can be created before the details of the
        beam_images are known.

        Args:
            beam_image (np.ndarray): An image of just the beam with no atoms,
                taken during absorption imaging. This should be a 2D array.
        """
        self.n_pixels = beam_image.size
        self.image_shape = beam_image.shape
        self.beam_image_hashes = []
        self.next_beam_image_index = 0
        self.n_beam_images = 0

        # Preallocate entire array to avoid re-allocating as beam_images are
        # added. Each column will contain the data from one image.
        self.beam_images = np.empty((self.n_pixels, self.max_beam_images))

        # Use entire image for PCA by default.
        self.set_mask(np.ones(self.image_shape))

        self._initialised = True

    @property
    def max_beam_images(self):
        """The maximum number of beam images to use as inputs to the PCA.

        Note that this also sets the size of a preallocated array used to store
        the data from these images. This array can take up a lot of memory if
        max_beam_images is set to a large value.
        """
        return self._max_beam_images

    @property
    def max_principal_components(self):
        """The max number of principal components to keep after doing PCA.

        Note that fewer may be used if there are not enough beam images to
        construct this many principal components.
        """
        return self._max_principal_components

    @max_principal_components.setter
    def max_principal_components(self, value):
        if value == self._max_principal_components:
            # Value hasn't changed, so don't need to do anything.
            pass
        elif value < self._max_principal_components and self.cache_valid:
            # We can take a subset of the cached principal components.
            mean_vector, principal_components, variances = self.pca_results
            principal_components = principal_components[:, :value]
            variances = variances[:value]
            self.pca_results = mean_vector, principal_components, variances
        else:
            # We need to compute more components.
            self._mark_cache_invalid()
        self._max_principal_components = value

    @property
    def use_sparse_routines(self):
        """Sets whether scipy's routines for dense or sparse matrices are used.

        Surprisingly, the routine for calculating eigenvectors of sparse
        matrices can be faster than the routine for dense matrices, even when
        the matrix is desnse. This may have to do with the fact that we don't
        always calculate all of the eigenvectors.
        """
        return self._use_sparse_routines

    @use_sparse_routines.setter
    def use_sparse_routines(self, value):
        # Only mark cache invalid if this property's value changes.
        if value != self._use_sparse_routines:
            self._use_sparse_routines = value
            self._mark_cache_invalid()

    def _mark_cache_invalid(self):
        """Mark that PCA analysis needs to be rerun.

        This method also clears the results from the previous PCA."""
        self.cache_valid = False
        self.pca_results = None

    @staticmethod
    def _image_to_vector(image):
        """Convert an image into a vector.

        This function takes an image, which is usually a 2D array of ints and
        converts it into a 1D array of floats. That format is what is necessary
        for this class's numerical routines.

        Args:
            image (numpy.ndarray): An image for use in the absorption image
                analysis.
        """
        return image.flatten().astype(float)

    def add_beam_image(self, beam_image, enable_image_shape_error=True):
        """Add a beam image to the list of images used for reconstruction.

        The first beam_image added can have any shape, but subsequent ones must
        all have the same shape as the first one. When this method is called
        with a beam_image of the wrong shape, the beam_image is not added. If
        enable_image_shape_error is True, then a ValuError will be raised if a
        beam_image of the wrong shape is given. If enable_image_shape_error is
        False, then the beam_image will still not be added, but the method will
        exit silently.

        If the provided beam_image has already been added, then it will NOT be
        added again and this method will just silently return.

        Args:
            beam_image (np.ndarray): An image (a 2D array) of the beam used for
                absorption imaging, without any atoms present in the image.
            enable_image_shape_error (bool, optional): (Default = True) Set
                whether or not to raise a ValueError if beam_image is of the
                wrong shape.

        Raises:
            ValueError: When beam_image is of a different shape than previously
                added beam images and enable_image_shape_error is True.
        """
        if not self._initialised:
            self._init(beam_image)
        else:
            # Check that this image is the same shape as the previous ones.
            if beam_image.shape != self.image_shape:
                if enable_image_shape_error:
                    error_message = (f"Beam image has shape {beam_image.shape} "
                                     f"but should have shape "
                                     f"{self.image_shape}.")
                    raise ValueError(error_message)
                else:
                    # In this case silently return without adding image.
                    return

        # Hash the image to check for uniqueness.
        image_hash = hash(beam_image.tobytes())
        if image_hash in self.beam_image_hashes:
            # Ignore duplicate image.
            return

        # Add new image, overwriting oldest one if max_beam_images is reached.
        if self.n_beam_images < self.max_beam_images:
            self.beam_image_hashes.append(image_hash)
        else:
            self.beam_image_hashes[self.next_beam_image_index] = image_hash
        self.beam_images[:, self.next_beam_image_index] = self._image_to_vector(
            beam_image)
        self.n_beam_images = len(self.beam_image_hashes)

        # Move along index for where the next reference image will go.
        self.next_beam_image_index += 1
        # Wrap around to overwrite oldest images.
        self.next_beam_image_index = int(
            self.next_beam_image_index %
            self.max_beam_images)

        # PCA now needs to be rerun.
        self._mark_cache_invalid()

    def add_beam_images(self, beam_images, **kwargs):
        """Convenience function to add many beam images.

        This function simply calls add_beam_image() for each image in
        beam_images.

        Args:
            beam_images (np.ndarray or list of np.ndarrays): This should be an
                iterable object (e.g. list or array) containing many beam
                images, each of which is a 2D numpy array. Therefore beam_images
                shoule be a list of 2D arrays or a 3D array such that
                beam_images[0] gives the first image, and so on.
            **kwargs: Keyword arguments are passed on to add_beam_image(). This
                allows setting that function's optional enable_image_shape_error
                argument.
        """
        for beam_image in beam_images:
            self.add_beam_image(beam_image, **kwargs)

    def set_mask(self, mask):
        """Set the mask that tells the analysis where the atoms may be.

        If the mask is not set manually, then the entire image is used.

        Args:
            mask (np.ndarray): The mask should be a 2D array with the same shape
                as the images. All of its entries should be 1 or 0. The 1's
                correspond to pixels that should be considered background region
                and used in the reconstruction. The 0's correspond to pixels
                which may have atoms, and so should be reconstructed using the
                other pixels.
        """
        self.mask = self._image_to_vector(mask)

        # PCA needs to be rerun.
        self._mark_cache_invalid()

    def set_rectangular_mask(self, atom_region_rows, atom_region_cols):
        """A convenience function to mark rectangular region as the atom region.

        An arbitrary mask can be specified using the set_mask() method. This
        function is included for convenience only.

        Args:
            atom_region_rows (list of ints): The indices specifying the region
                which may have atoms. This should be a list or tuple of two
                elements, and each element should be an integer. All the rows
                between atom_region_rows[0] and atom_region_rows[1] will be
                considered part of the atom region. Note that as is usual with
                python indexing, the row specified by the lower index will be
                included but the row specified by the upper index will not. So
                for example, setting atom_region_rows=[0,3] will mark rows
                0, 1, and 2 as the atom region. Note that this argument marks
                which rows are considered atom region, which sets the atom
                region in the vertical direction of the image.
            atom_region_cols (list of ints): The same as atom_region_rows,
                except it sets the corresponding values for the column indices,
                thereby controlling the horizontal direction of the atom region.
        """
        # Check that indices are sane.
        n_rows, n_cols = self.image_shape
        for i in range(len(atom_region_rows)):
            if abs(atom_region_rows[i]) > n_rows:
                error_message = (f"atom_region_rows[{i}] has value "
                                 f"{atom_region_rows[i]} but should be <={n_rows} "
                                 f"and >={-n_rows}.")
                raise IndexError(error_message)
        for i in range(len(atom_region_cols)):
            if abs(atom_region_cols[i]) > n_cols:
                error_message = (f"atom_region_cols[{i}] has value "
                                 f"{atom_region_cols[i]} but should be <={n_cols} "
                                 f"and >={-n_cols}.")
                raise IndexError(error_message)

        # Construct the mask and store it.
        mask = np.ones(self.image_shape)
        mask[atom_region_rows[0]:atom_region_rows[1],
             atom_region_cols[0]:atom_region_cols[1]] = 0
        self.set_mask(mask)

    def get_pca_images(self):
        """Return mean_beam, mask, principal_component_images, and variances.

        Images will be reshaped back from vectors into 2D arrays. The principal
        component with the largest variance will be
        principal_component_images[0], and the one with the second largest will
        be principal_component_images[1], and so on.
        """
        mean_vector, principal_components, variances = self.pca()
        shape = self.image_shape + (principal_components.shape[1],)
        principal_component_images = principal_components.reshape(shape)
        principal_component_images = np.moveaxis(
            principal_component_images, -1, 0)
        mean_beam = mean_vector.reshape(self.image_shape)
        mask = self.mask.reshape(self.image_shape)
        return mean_beam, mask, principal_component_images, variances

    def get_pca_mean_beam(self):
        """Get the mean beam image calculated during PCA.

        Will be returned as a 2D array. Note that calling this method will run
        the PCA analysis if it has not yet been run.
        """
        return self.get_pca_images()[0]

    def get_pca_mask(self):
        """Get the mask used during PCA..

        Will be returned as a 2D array. Note that calling this method will run
        the PCA analysis if it has not yet been run.
        """
        return self.get_pca_images()[1]

    def get_pca_principal_component_images(self):
        """Get the principal component images calculated during PCA.

        Will be returned as an array of 2D arrays. The principal component with
        the largest variance will be principal_component_images[0], and the one
        with the second largest will be principal_component_images[1], and so
        on. Note that calling this method will run the PCA analysis if it has
        not yet been run.
        """
        return self.get_pca_images()[2]

    def get_pca_principal_component_image(self, component_index):
        """Get a principal component image calculated during PCA.

        Will be returned as a 2D array. If component_index is 0, then the
        principal component with the largest variance will be returned. If
        component_index is 1, then the principal component with the second
        largest variance will be returned, and so on. Note that calling this
        method will run the PCA analysis if it has not yet been run.

        Args:
            component_index (int): The index of the principal component to
                display. The princicpal components are numbered in order from
                largest associated variance to lowest, starting at 0.

        Raises:
            ValueError: If component_index isn't a a valid index, i.e. if it
                isn't an integer or if it is less than zero, or greater than or
                equal to the number of principal components.

        Returns:
            pca_image (array): The requested principal component, as a 2D array.
        """
        # Check that component_index is an integer between zero and the number
        # of principal components available.
        principal_components = self.get_pca_principal_component_images()
        n_components = len(principal_components)
        if not isinstance(component_index, int):
            message = ("component_index must be an integer but is "
                       f"{component_index}")
            raise ValueError(message)
        if component_index < 0 or component_index >= n_components:
            message = ("component_index must be between 0 and "
                       f"{n_components-1}.")
            raise ValueError(message)

        # Retrieve the image.
        pca_images = self.get_pca_principal_component_images()
        pca_image = pca_images[component_index]

        return pca_image

    def get_pca_variances(self):
        """Get the variances calculated during PCA.

        Will be returned as a 1D array. These variances are the variances
        associated with the principal components, ordered from largest variance
        to smallest. This is the same ordering as for the princial components
        themselves returned by get_pca_principal_component_images(), so the ith
        image there is the principal component with variance stored as the ith
        entry in the array returned by this method.
        """
        return self.get_pca_images()[3]

    def pca(self):
        """Get principal component analysis results.

        This function will return cached results if they are valid, or rerun the
        PCA if necessary.

        This function uses the same algorithm as the MATLAB code here:
        https://github.com/zakv/AbsorptionImageProcessing_MATLAB, so see the
        README in that repository for additional information.

        Raises:
            RuntimeError: If no reference images are available and no data has
                been loaded with load_pca()

        Returns:
            A tuple (mean_vector, principal_components, variances). Each entry
                in the tuple is a numpy array. The arrays with images have the
                images stored as columns rather than 2D arrays. For the end user
                it is better to access that data with the corresponding
                get_pca_... methods, which will make sure that the data is
                reshaped to 2D arrays where appropriate.
        """
        if not self._initialised and not self.cache_valid:
            msg = "No reference images added or previously computed PCA basis loaded"
            raise RuntimeError(msg)
        if not self.cache_valid:
            self.pca_results = self._pca()
            self.cache_valid = True
        return self.pca_results

    def _pca(self):
        """Perform the principal component analysis.

        Do not call this function directly. Instead use pca() (without the
        leading underscore), which will return cached results if they are
        available.
        """
        mean_beam = np.mean(self.beam_images, axis=1, keepdims=False)
        mask = self.mask
        beam_images = self.beam_images[:, :self.n_beam_images]

        # Subtract mean_beam from images and apply the mask. Element-wise
        # multiplication and subtraction using numpy broadcasting (as commented
        # out below) requires 3 large matrices in memory at an intermediate
        # point in the computation, namely right after (beam_images -
        # mean_beam_2d) is evaluated and memory for centered_masked_images is
        # allocated.
        # mask_2d = mask[:,np.newaxis]
        # mean_beam_2d = mean_beam[:,np.newaxis]
        # centered_masked_images = mask_2d * (beam_images - mean_beam_2d)

        # Instead of that direct approach, use self._center_and_mask_numba() or
        # self._center_and_mask_in_place(). As of this writing the _in_place
        # version is faster, but this may change in the future since the numba
        # version supports parallelization.
        centered_masked_images = self._center_and_mask_in_place(
            beam_images,
            mask,
            mean_beam,
        )
        # centered_masked_images should be C-contiguous already but it's good to
        # make sure.
        centered_masked_images = np.ascontiguousarray(centered_masked_images)

        # Compute the masked principal components.
        # The -1 is there because last eigenvector isn't necessarily orthogonal
        # to the others.
        n_eigs = min(self.n_beam_images - 1, self.max_principal_components)
        n_eigs = max(n_eigs, 1)  # Need at least one.
        # .T means transpose, @ means matrix multiplication.
        cov_mat = centered_masked_images.T @ centered_masked_images
        del centered_masked_images  # Free up memory.
        if self.use_sparse_routines:
            variances, principal_components = eigsh(
                cov_mat, k=n_eigs, which='LM')
        else:
            eigvals_param = (
                self.n_beam_images - n_eigs,
                self.n_beam_images - 1,
            )
            variances, principal_components = eigh(
                cov_mat,
                eigvals=eigvals_param,
                overwrite_a=True,  # overwrite_a might reduce memory usage.
            )
        del cov_mat  # Free up memory.

        # Reverse ordering to put largest eigenvectors/eigenvalues first.
        principal_components = np.fliplr(principal_components)
        variances = np.flip(variances)

        # principal_components isn't always C-contiguous, and when it's not the
        # matrix multiplication below becomes extremely slow. It's much faster
        # to make it C-contiguous first so that numpy can use faster matrix
        # multiplication routines behind the scenes.
        principal_components = np.ascontiguousarray(principal_components)

        # Construct the un-masked basis vectors.
        centered_images = beam_images - mean_beam[:, np.newaxis]
        # centered_images should be C-contiguous already but it's good to make
        # sure.
        centered_images = np.ascontiguousarray(centered_images)
        principal_components = centered_images @ principal_components
        del centered_images  # Free up memory.

        # As of this writing, self._normalize_vectorized() is faster than using
        # self._normalize_numba() despite the fact that the latter is uses numba
        # and allows for parallelization. That may change in the future though.
        principal_components = self._normalize_vectorized(
            principal_components,
            mask,
        )

        return mean_beam, principal_components, variances

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _center_and_mask_numba(beam_images, mask, mean_beam):
        # The code below does mask * (beam_image - mean_beam) but with better
        # memory management at the cost of using a for loop. Using numba speeds
        # up this for loop, although, at least as of this writing, it still
        # isn't as fast as the memory-hogging approach or the
        # _center_and_mask_in_place() approach. That may change in the future
        # though, especially given that this should be parallelized.
        centered_masked_images = np.empty_like(beam_images)
        for i in prange(centered_masked_images.shape[1]):
            beam_image = beam_images[:, i]
            centered_masked_images[:, i] = mask * (beam_image - mean_beam)
        return centered_masked_images

    @staticmethod
    def _center_and_mask_in_place(beam_images, mask, mean_beam):
        # This is defined as a static method so that it has the same arguments
        # as _center_and_mask_numba() which makes it quick/easy to switch
        # back/forth between them.

        # The following calculates mask * (beam_image - mean_beam), but does so
        # in place so reduce memory usage during the calculation.
        centered_masked_images = beam_images.copy()
        centered_masked_images -= mean_beam[:, np.newaxis]
        centered_masked_images *= mask[:, np.newaxis]
        return centered_masked_images

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _normalize_numba(principal_components, mask):
        # Normalize columns over the unmasked region.
        for i in prange(principal_components.shape[1]):
            column_norm = np.linalg.norm(mask * principal_components[:, i])
            principal_components[:, i] /= column_norm
        return principal_components

    @staticmethod
    def _normalize_vectorized(principal_components, mask):
        # This is defined as a static method so that it has the same arguments
        # as _normalize_numba() which makes it quick/easy to switch back/forth
        # between them.
        masked_components = mask[:, np.newaxis] * principal_components
        norms = np.linalg.norm(masked_components, axis=0)
        principal_components /= norms
        return principal_components

    def save_pca(self, filepath):
        """Save cached PCA results to disk.

        Args:
            filepath (str): The full name (including path and extension) that
                should be used to save the data. Data will be saved in numpy's
                .npy format, so that should be the file extension.
        """
        mean_beam, principal_components, variances = self.pca()
        image_shape = np.array(self.image_shape)
        with open(filepath, 'wb') as f:
            np.save(f, image_shape)
            np.save(f, mean_beam)
            np.save(f, principal_components)
            np.save(f, variances)
            np.save(f, self.mask)

    def load_pca(self, filepath):
        """Restore saved PCA results from disk.

        Since you may load any previously computed PCA basis, this may or may
        not be consistent with any reference images you have added (and you may
        have not added any reference images at all). It is up to you to keep
        track of this. If you add more reference images, the PCA basis will
        deleted and recomputed from the set of reference images. This could lead
        to subtle mistakes if you are not careful.

        Args:
            filepath (str): The full name (including path and extension) that
                of the file containing the PCA data. The file should be one that
                was created with this class's save_pca() method.
        """
        with open(filepath, 'rb') as f:
            image_shape = np.load(f)
            mean_beam = np.load(f)
            principal_components = np.load(f)
            variances = np.load(f)
            mask = np.load(f)

        image_shape = tuple(image_shape)
        if not self._initialised:
            self._init(np.empty(image_shape))
        elif self.image_shape != image_shape:
            msg = 'image shape does not match'
            raise ValueError(msg)
        self.set_mask(mask)
        self.pca_results = mean_beam, principal_components, variances

    def plot_mean_beam(self, *args, **kwargs):
        """Display the mean beam image as a false color plot.

        Args:
            *args: Additional arguments are passed to self.plot_image(). See
                that method's documentation for more information.
            **kwargs: Additional keyword arguments are passed to
                self.plot_image(). See that method's documentation for more
                information.

        Returns:
            axes (matplotlib.axes._subplots.AxesSubplot): The axes on which the
                plot was made.
        """
        # Get the principal components.
        mean_beam = self.get_pca_mean_beam()

        # Generate the plot
        axes = self.plot_image(mean_beam, *args, **kwargs)
        axes.set_title("Mean Beam Image")

        # Return the axes of the plot.
        return axes

    def plot_principal_component(self, component_index, *args, **kwargs):
        """Display a principal component as a false color plot.

        Args:
            component_index (int): The index of the principal component to
                display. The princicpal components are numbered in order from
                largest associated variance to lowest, starting at 0.
            *args: Additional arguments are passed to self.plot_image(). See
                that method's documentation for more information.
            **kwargs: Additional keyword arguments are passed to
                self.plot_image(). See that method's documentation for more
                information.

        Returns:
            axes (matplotlib.axes._subplots.AxesSubplot): The axes on which the
                plot was made.
        """
        # Get the principal component and associated variance.
        principal_component = self.get_pca_principal_component_image(
            component_index,
        )
        variances = self.get_pca_variances()
        variance = variances[component_index]

        # Generate the plot.
        axes = self.plot_image(principal_component, *args, **kwargs)
        axes.set_title(f"Principal Component {component_index}\n"
                       f"Variance: {variance:.2e}")

        # Return the axes of the plot.
        return axes

    def plot_image(self, image, axes=None, colorbar=True, imshow_args={}):
        """Display an image as a false color plot.

        Args:
            image (array): A 2D array representing an image to be plotted.
            axes (matplotlib.axes._subplots.AxesSubplot, optional):
                (Default=None) The axes on which the image should be plotted. If
                set to None, new axes on a new figure will be used.
            colorbar (bool, optional): (Default=True) Set whether or not a
                colorbar should be plotted.
            imshow_args (dict, optional): (Default={})  Keyword-style arguments
                can be passed to matplotlib's imshow function (the one that is
                used for the plot) by including them in this dictionary. Note
                that the name of the arguments should be specifed as strings, so
                this can be something like {'vmax':0.7}.

        Returns:
            axes (matplotlib.axes._subplots.AxesSubplot): The axes on which the
                plot was made.
        """
        # Set up new axes if necessary.
        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)

        # Plot the image.
        # Set options, without overwriting caller settings.
        imshow_args = {
            'X': image,
            'cmap': plt.get_cmap('jet'),
            **imshow_args
        }
        imshow_plot = axes.imshow(**imshow_args)
        # Make the ascpect ratio 1 to avoid stretching the image.
        axes.set_aspect(1.)

        # Add a colorbar if requested.
        if colorbar:
            plt.colorbar(imshow_plot)

        # Return the axes of the plot.
        return axes

    def reconstruct(self, atoms_image, return_coeffs=False):
        """Reconstruct an atoms_image as a sum of beam images.

        Note that since images are centered prior to PCA and reconstruction when
        using PCA, the difference between the mean image and the atoms_image is
        what is reconstructed using a linear sum of principal components, so the
        reconstructed image is:

        reconstructed_image = mean_image + \\sum_i coeff_i * pca_basis_vector_i

        Args:
            atoms_image (np.ndarray): A 2D numpy array of data representing an
                image of the atomic cloud taken for absorption imaging.
            return_coeffs (bool, optional): (Default = False) If set to true,
                the coefficients used to weight the principal components in the
                reconstruction will aslo be returned.

        Raises:
            RuntimeError: If no beam images have been added and no PCA basis has
                been loaded from a file.
            ValueError: If the image doesn't have the same shape as the beam
                images used for the PCA.

        Returns:
            reconstructed_image: The reconstructed image (a 2D array of floats
                with the same shape as image) representing what the image would
                have looked like without atoms.
            coefficients: A numpy array of the coefficients used to weight the
                principal components in the reconstruction. This is only
                returned if return_coeffs is set to True.
        """
        if not self._initialised and not self.cache_valid:
            msg = "No beam images added or previously computed PCA basis loaded"
            raise RuntimeError(msg)

        # Ensure that image has the correct shape.
        if atoms_image.shape != self.image_shape:
            error_message = (f"Beam image has shape {atoms_image.shape} "
                             f"but should have shape {self.image_shape}.")
            raise ValueError(error_message)

        # Get the PCA results.
        mean_beam, principal_components, _ = self.pca()

        # Calculate weights and use them to reconstruct the image.
        masked_image = self.mask * \
            (self._image_to_vector(atoms_image) - mean_beam)
        coefficients = principal_components.T @ masked_image
        reconstructed_image = (principal_components @ coefficients) + mean_beam

        # Reshape the reconstruction to the same shape as the input.
        reconstructed_image = reconstructed_image.reshape(atoms_image.shape)

        if return_coeffs:
            return reconstructed_image, coefficients
        else:
            return reconstructed_image

    def get_od_image(self, atoms_image):
        """Use PCA to caculate the optical depth of a cloud.

        Args:
            atoms_image (np.ndarray): A 2D numpy array of data representing an
                image of the atomic cloud taken for absorption imaging.

        Returns:
            od_image (np.ndarray): A 2D array of floats which give an image of
                the cloud's optical depth.
        """
        reconstruction = self.reconstruct(atoms_image)
        od_image = np.log(reconstruction / atoms_image)

        return od_image
