import numpy as np

from primitives.track import Track

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

class PixelCoordinateTransform(Transform):
	""" Changes coordinates so origin is in the bottom left corner

	"""

	def __init__(self, imgSize):
		self._imgWidth, self._imgHeight = imgSize

	def transformPoints(self, points):
		newPoints = [np.array([x, self._imgHeight - y]) for x,y in points]

		return newPoints

	def transformTrack(self, track):
		points = np.asarray(track.positions)
		undistorted = [np.array([x, self._imgHeight - y]) for x,y in track.positions]
		newTrack = Track(np.asarray(undistorted), track.times, track.id)
		return newTrack

	def transformTracks(self, tracks):
		for track in tracks:
			points = np.asarray(track.positions)
			undistorted = [np.array([x, self._imgHeight - y]) for x,y in track.positions]
			newTrack = Track(np.asarray(undistorted), track.times, track.id)
			yield newTrack

	def transformImage(self, img):
		newImg = np.flipud(img[:,:,:])

		return newImg