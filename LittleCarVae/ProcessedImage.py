import numpy as np

class ProcessedImage(object):
	def __init__(self, fname):
		self.filename = fname
		self.cache = {}
		self.original_array = None

	def get_original(self):
		# load the image, if not already
		if not self.original_array:
			self.original_array = np.array(10)
		return self.original_array

	def get_processed_image(self, sz, color_depth):
		processed_array, _ = self.cache.get((sz,color_depth))
		if not processed_array:

			original_array = self.get_original()

			# crop the image

			# resize the image

			# whiten the image
			prestats = 0.0, 0.0
			processed_array = np.array(10)

			# add to the cache
			self.cache[sz,color_depth] = processed_array, prestats

		return processed_array

	def reconstruct_from_predicted(self, predicted):
		mean, std = self.prestats
		return np.array(4,4,2)

	def render_image(self, ax, array=None):
		pass

	@static_method
	def from_dir(dirname):
		pass

