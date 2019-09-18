import os
from enum import Enum
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.util import crop, pad

class ColorModel(Enum):
    RGB = 1
    HSV = 2
    GRAY = 3

class ProcessedImage(object):
	def __init__(self, fullpath):
		self.fullpath = fullpath
		self.cache = {}
		self.original_img = None

	def __str__(self):
		self_str = 'ProcessedImage for {} ({} in cache)'
		return self_str.format(self.fullpath, len(self.cache))

	def get_original(self):
		""" load the original image, if not already """
		if self.original_img is None:
			self.original_img = mpimg.imread(self.fullpath)
		return self.original_img

	def get_processed_image(self, sz=64, color_model=ColorModel.GRAY, augment_imgs=False):
		""" gets a processed version of the image, if not in cache """
		if not (sz,color_model) in self.cache:
			img = self.get_original()
			
			if augment_imgs:   # crop the image, if augment requested
				# TODO: generate margins randomly?
				img = crop(img, ((5,0),(3,2),(0,0)))
			
			img = resize(img, (sz,sz), anti_aliasing=True)

			if color_model == ColorModel.GRAY: 
				img = img[...,0]

			# whiten the image
			mean, std = img[:][:][:][0].mean(), img[:][:][:][0].std()
			img = img - mean
			img = img / std

			# add to the cache
			self.cache[(sz,color_model)] = img, (mean,std)
			return img
		else:
			img, _ = self.cache[(sz,color_model)]
			return img

	def reconstruct_from_predicted(self, predicted, color_model=ColorModel.GRAY):
		sz = predicted.shape[0]
		_, (mean, std) = self.cache[sz,color_model]
		reconst = predicted
		reconst = reconst * std
		reconst = reconst - mean
		return reconst

	@staticmethod
	def image_strip(imgs, axes, predicted=None):
		""" shows a collection of images, and possibly their reconstruction, on a figure axes """
		for n in range(axes.shape[-1]):
			img = imgs[n].get_processed_image()
			if (len(axes.shape) == 1):
				axis = axes[n]
			else:
				axis = axes[0,n]
			axis.imshow(img, cmap='gray')
			if not predicted is None:
				reconst = img.reconstruct_from_predicted(predicted[imgs[n].fullpath])
				axes[1][n].imshow(reconst)

	@staticmethod
	def from_dir(dirname):
		""" returns a collection of images from a directory to be searched """
		for path, _, files in os.walk(dirname): 
			for file in files:
				yield ProcessedImage(os.path.join(path,file)) 

if __name__ == '__main__':
	processed_imgs = ProcessedImage.from_dir('..\\Data')
	for img in processed_imgs:
		print(img)
		print('Processed image shape = {}'.format(img.get_processed_image().shape))