import numpy as np
import cv2

from primitives.track import Track

from .base import Transform
from .. import cams

class UndistortionTransform(Transform):

	def __init__(self, camera):
		self._camera = camera

	def transformPoints(self, points):
		return self._camera.undistortPoints(points)

	def inverseTransformPoints(self, points):
		return self._camera.distortPoints(points)

	def transformTrack(self, track):
		points = np.asarray(track.positions)
		undistorted = self._camera.undistortPoints(points)
		newTrack = Track(np.asarray(undistorted), track.times, track.id)
		return newTrack

	def transformTracks(self, tracks):
		for track in tracks:
			points = np.asarray(track.positions)
			undistorted = self._camera.undistortPoints(points)
			newTrack = Track(np.asarray(undistorted), track.times, track.id)
			yield newTrack
			
	def inverseTransformTrack(self, track):
		points = np.asarray(track.positions)
		distorted = self._camera.distortPoints(points)
		newTrack = Track(np.asarray(distorted), track.times, track.id)
		return newTrack

	def transformImage(self, img):
		return self._camera.undistortImage(img)

class ExtrinsicsTransform(Transform):
	""" For now this just rotates the frame to fix coordinates for plotting

	"""

	def __init__(self, camera):
		self._camera = camera
		width, height = camera.imgSize
		self._rotation = cv2.getRotationMatrix2D((width/2,height/2),180,1)


	def transformPoints(self, points):


		return points

	def transformTrack(self, track):
		points = np.asarray(track.positions)
		undistorted = [np.dot(self._rotation, np.append(p, [1])) for p in track.positions]
		newTrack = Track(np.asarray(undistorted), track.times, track.id)

		return newTrack

	def transformImage(self, img):
		rows,cols = img.shape[:2]
		newImg = cv2.warpAffine(img, self._rotation, (cols,rows))
		return newImg

