
class ProcessedImage(object):
	def __init__(self, fname, sz, color_depth):
		self.filename = fname
		self.processed_size = sz
		self.processed_depth = color_depth
	
	def process_image(self):
		# load the image

		# crop the image

		# resize the image

		# whiten the image
		self.prestats = 0.0, 0.0

		import numpy as np
		self.processed_array = np.array(10)

	def reconstruct_from_predicted(self, predicted):
		mean, std = self.prestats

		import numpy as np
		return np.array(4,4,2)

	def render_image(self, ax, array=None):
		pass
