import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.ion()

class ImageView(object):
	
	def __init__(self, img=None):
		if (img is not None):
			# Convert img from BGR (OpenCV Colors) to RGB
			self._img = img[:, :, ::-1]
			# Flip image so origin is in lower left corner 
			self._img = np.flipud(self._img)

	def plot(self, pause=0.001):
		plt.imshow(self._img, origin='lower')
		plt.show()
		plt.pause(pause)