import numpy as np
import cv2

class LKOpticalFlowTracker(object):

	def __init__(self, winSize, maxLevel, criteria, firstImage=None):
		self._winSize = winSize
		self._maxLevel = maxLevel
		self._criteria = criteria

		self._prevImg = firstImage

		self._deviationThreshold = 1

	def trackPoints(self, points, img):
		if (self._prevImg is None):
			self._prevImg = img
			return [None for _ in points]

		self._params = dict(winSize  = self._winSize, maxLevel = self._maxLevel,
							criteria = self._criteria)

		# Run LK forwards
		nextPoints, status, error = cv2.calcOpticalFlowPyrLK(self._prevImg, img, 
			points, None, **self._params)

		# Run LK in reverse
		prevPoints, status, error = cv2.calcOpticalFlowPyrLK(img, self._prevImg,
			nextPoints, None, **self._params)

		# Compute deviation between original points and back propagation
		dev = abs(points - prevPoints).reshape(-1,2)

		# For each pair of points, take max deviation in either axis
		maxDev = dev.max(-1)

		# Check against max deviation threshold allowed
		matchQuality = maxDev < self._deviationThreshold

		trackedPoints = []

		for (point, match) in zip(nextPoints.reshape(-1,2), matchQuality):
			if (match):
				trackedPoints.append(point)
			
		self._prevImg = img

		return np.float32(trackedPoints).reshape(-1, 2)