import numpy as np
import cv2

class LKOpticalFlowTracker(object):

	def __init__(self, winSize, maxLevel, criteria, firstImage=None):
		self._winSize = winSize
		self._maxLevel = maxLevel
		self._criteria = criteria

		self._prevImg = firstImage

		self._deviationThreshold = 1


		self._params = dict(winSize = self._winSize, maxLevel = self._maxLevel,
							criteria = self._criteria)

	def trackPoints(self, points, img):
		if (self._prevImg is None):
			self._prevImg = img
			return [None for _ in points]


		# Run LK forwards
		nextPoints, status, error = cv2.calcOpticalFlowPyrLK(self._prevImg, img, 
			points, None, **self._params)

		del status, error

		# Run LK in reverse
		prevPoints, status, error = cv2.calcOpticalFlowPyrLK(img, self._prevImg,
			nextPoints, None, **self._params)

		del self._prevImg, status, error

		# Compute deviation between original points and back propagation
		dev = abs(points - prevPoints).reshape(-1,2)

		# For each pair of points, take max deviation in either axis
		maxDev = dev.max(-1)

		del dev

		# Check against max deviation threshold allowed
		matchQuality = maxDev < self._deviationThreshold

		trackedPoints = []

		for (point, match) in zip(nextPoints.reshape(-1,2), matchQuality):
			if (match):
				trackedPoints.append(point)
			else:
				trackedPoints.append(None)
			
		del matchQuality, points, nextPoints, prevPoints
		self._prevImg = img

		return np.asarray(trackedPoints)
		#np.float32(trackedPoints).reshape(-1, 2)