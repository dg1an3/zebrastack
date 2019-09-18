import numpy as np
from enum import Enum

class ColorModel(Enum):
    RGB = 1
    HSV = 2
    GRAY = 3

class ProcessedImage(object):
	def __init__(self, fullpath):
		self.fullpath = fullpath
		self.cache = {}
		self.original_img = None

	def get_original(self):
		# load the image, if not already
		if not self.original_img:
			import matplotlib.image as mpimg
			self.original_img = mpimg.imread(fullpath)
		return self.original_img

	def get_processed_image(self, sz=64, color_model=ColorModel.GRAY, augment_imgs=False):
		img, _ = self.cache.get((sz,color_model))
		if not processed_array:
			from skimage.transform import resize, crop
			img = self.get_original()

			# crop the image
			if augment_imgs:
				# TODO: generate margins randomly?
				img = crop(img, ((5,0),(3,2),(0,0)))

			# resize the image
			img = resize(img, (128,128), anti_aliasing=True)

			# whiten the image
			mean, std = img[:][:][:][0].mean(), img[:][:][:][0].std()
			img = img - mean
			img = img / std

			# add to the cache
			self.cache[sz,color_model] = processed_array, (mean,std)
		return img

	def reconstruct_from_predicted(self, predicted, color_model=ColorModel.GRAY):
		sz = predicted.shape[0]
		_, (mean, std) = self.cache[sz,color_model]
		reconst = predicted
		reconst = reconst * std
		reconst = reconst - mean
		return reconst

	def render_image(self, ax, array=None):
		pass

	@staticmethod
	def from_dir(dirname):
		import os
		for path, _, files in os.walk(dirname): 
			for file in files:
				yield ProcessedImage(os.path.combine(path,file)) 
