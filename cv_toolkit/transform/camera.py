import numpy as np

from primitives.track import Track

from .base import Transform
from .. import cams

class UndistortionTransform(Transform):

	def __init__(self, camera):
		self._camera = camera

	def transformPoints(self, points):
		return self._camera.undistortPoints(points)

	def transformTrack(self, track):
		points = np.asarray(track.positions)
		undistorted = self._camera.undistortPoints(points)
		newTrack = Track(np.asarray(undistorted), track.times)
		return newTrack

	def transformImage(self, img):
		return self._camera.undistortImage(img)
