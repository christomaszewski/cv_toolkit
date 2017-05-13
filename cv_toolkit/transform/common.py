import numpy as np

from .base import Transform

class IdentityTransform(Transform):

	def __init__(self):
		# Do nothing special
		pass

	def transformPoints(self, points):
		return points

	def transformTrack(self, track):
		return track

	def transformImage(self, img):
		return img