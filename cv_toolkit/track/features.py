import numpy as np
import cv2

from primitives.keypoint import KeyPoint

from memory_profiler import profile

class KeyPointWrapper(object):

	def __init__(self, point):
		self.pt = tuple(point)

class SparseFeatureTracker(object):

	def __init__(self, detector):
		self._detector = detector
		self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		self._prevImg = None

	def initialize(self, img, mask):
		self._kp, self._descriptors = self._detector.detectKeyPoints(img, mask)
		self._prevImg = img

		return self._kp

	@profile
	def trackPoints(self, points, img, mask=None):
		if (self._prevImg is None):
			self._prevImg = img
			return

		self._kp = [cv2.KeyPoint(**(kp.getCVKeyPointParams())) for kp in points if kp is not None]
		self._descriptors = self._detector.compute(self._prevImg, self._kp)
	
		keyPoints, descriptors = self._detector.detectKeyPoints(img, mask)

		matches = self._matcher.match(self._descriptors, descriptors)
		print(len(keyPoints), len(self._kp), len(matches))
		matchedKP = [keyPoints[m.trainIdx] for m in matches]
		indices = [m.queryIdx for m in matches]

		#print(matchedKP)
		self._prevImg = img

		self._kp = keyPoints
		self._descriptors = descriptors

		return (matchedKP, indices)

