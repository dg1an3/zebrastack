""" processed_image module
represents an original data image, and the logic for pre-processing and making
it ready for training.  also caches the different options.
Eventually, may wrap up persistence cached versions of the image
"""

import os
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.util import crop

class ColorModel(Enum):
    """Enum that represents different color models."""
    RGB = 1
    HSV = 2
    GRAY = 3

class ProcessedImage:
    """Represents a single original image, and the processed versions of same."""
    def __init__(self, fullpath):
        self.fullpath = fullpath
        self.cache = {}
        self.original_img = None

    def __str__(self):
        self_str = 'ProcessedImage for {} ({} in cache)'
        return self_str.format(self.fullpath, len(self.cache))

    def get_original(self):
        """Load the original image, if not already loaded."""
        if self.original_img is None:
            self.original_img = mpimg.imread(self.fullpath)
        return self.original_img

    def get_processed_image(self, size=64, color_model=ColorModel.GRAY, augment_imgs=False):
        """Gets a processed version of the image, if not in cache."""
        if not (size, color_model) in self.cache:
            img = self.get_original()

            if augment_imgs:   # crop the image, if augment requested
                # TODO: generate margins randomly?
                img = crop(img, ((5, 0), (3, 2), (0, 0)))

            img = resize(img, (size, size), anti_aliasing=True)

            mean, std = img.mean(), img.std()
            if color_model == ColorModel.GRAY:
                img = img[..., 0]                
                img = np.clip(0.5 + (img - mean)/(2.0*std), 0.0, 1.0)
            else:
                throw_exception()
            self.cache[(size, color_model)] = img, (mean, std)
        else:
            img, _ = self.cache[(size, color_model)]
        return img

    def reconstruct_from_predicted(self, predicted, color_model=ColorModel.GRAY):
        size = predicted.shape[0]
        _, stats = self.cache[size, color_model]
        mean, std = stats
        return (mean + predicted) * 2.0 * std

def show_image_strip(imgs, axes, predicted_dict=None):
    """Shows a collection of images, and possibly their reconstruction, on a figure axes."""
    for n in range(axes.shape[-1]):
        img = imgs[n].get_processed_image()
        if len(axes.shape) == 1:
            axis = axes[n]
        else:
            axis = axes[0, n]
        axis.imshow(img, cmap='gray')
        if not predicted_dict is None:
            predicted = predicted_dict[imgs[n].fullpath]
            reconst = imgs[n].reconstruct_from_predicted(predicted)
            axes[1][n].imshow(reconst, cmap='gray')

def read_from_dir(dirname):
    """Collection of images from a directory to be searched."""
    for path, _, files in os.walk(dirname):
        for file in files:
            yield ProcessedImage(os.path.join(path, file))

if __name__ == '__main__':
    processed_imgs = list(read_from_dir('..\\Data\\LittleCarDb1'))
    num_thumbnails = 3
    _, axes_orig = plt.subplots(1, num_thumbnails, sharey=True, figsize=(2, num_thumbnails))
    show_image_strip(processed_imgs, axes_orig)
    plt.show()

    predicted = {img.fullpath:img.get_processed_image() for img in processed_imgs}
    _, axes_reconst = plt.subplots(2, num_thumbnails, sharey=True, figsize=(2, num_thumbnails))
    show_image_strip(processed_imgs, axes_reconst, predicted)
    plt.show()
