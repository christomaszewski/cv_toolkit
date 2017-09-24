import cv2
import numpy as np

from .base import Detector

class ShiTomasiDetector(Detector):

	def __init__(self, maxCorners=100, qualityLevel=0.3, minDistance=10., blockSize=10):
		self._featureLimit = maxCorners
		self._qualityLevel = qualityLevel
		self._minDistance = minDistance
		self._blockSize = blockSize

		self._params = dict(maxCorners = maxCorners,
					  qualityLevel = qualityLevel,
					  minDistance = minDistance,
					  blockSize = blockSize)

		self._detector = cv2.goodFeaturesToTrack

	def detect(self, img, mask=None):
		# Should store current mask for reuse?

		return self._detector(img, mask=mask, **self._params)

	@property
	def featureLimit(self):
		return self._featureLimit

	@featureLimit.setter
	def featureLimit(self, limit):
		self._featureLimit = limit


# still needs to be implemented
class ORBDetector(Detector):

	def __init__(self, params):
		# set params etc
		self._params = params
		self._featureLimit = params['nfeatures']
		self._detector = cv2.ORB_create(**params)
		

	def detect(self, img, mask=None):
		keyPoints = self._detector.detect(img, mask)
		points = np.asarray([kp.pt for kp in keyPoints], dtype='float32')
		return points

	def detectKeyPoints(self, img, mask=None):
		keyPoints, descriptors = self._detector.detectAndCompute(img, mask)

		return (keyPoints, descriptors)

	def detectAndCompute(self, img, mask=None):
		keyPoints, descriptors = self._detector.detectAndCompute(img, mask)

		points = np.asarray([kp.pt for kp in keyPoints], dtype='float32')

		return (points, descriptors)

	def compute(self, img, keyPoints):
		keyPoints, descriptors = self._detector.compute(img, keyPoints)

		return descriptors

	@property
	def featureLimit(self):
		return self._featureLimit

	@featureLimit.setter
	def featureLimit(self, limit):
		self._params['nfeatures'] = limit
		self._featureLimit = limit
		self._detector = cv2.ORB_create(**self._params)
