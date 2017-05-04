import cv2
import numpy as np

from abc import ABCMeta, abstractmethod

class Detector(metaclass=ABCMeta):

	@abstractmethod
	def detect(self, img, mask):
		pass

	@property
	@abstractmethod
	def featureLimit(self):
		pass

	@featureLimit.setter
	@abstractmethod
	def featureLimit(self, limit):
		pass


class ShiTomasiDetector(Detector):

	def __init__(self, maxCorners=100, qualityLevel=0.9, minDistance=10, blockSize=7):
		self._featureLimit = maxCorners
		self._qualityLevel = qualityLevel
		self._minDistance = minDistance
		self._blockSize = blockSize

		self._detector = cv2.goodFeaturesToTrack

	def detect(self, img, mask):
		params = dict(maxCorners = self._featureLimit,
					  qualityLevel = self._qualityLevel,
					  minDistance = self._minDistance,
					  blockSize = self._blockSize)

		return self._detector(img, mask=mask, **params)

	@property
	def featureLimit(self):
		return self._featureLimit

	@featureLimit.setter
	def featureLimit(self, limit):
		self._featureLimit = limit


class GridDetector(Detector):

	@classmethod
	def using_shi_tomasi(cls, gridDim, featureLimit, borderBuffer=0):
		detector = ShiTomasiDetector()
		return cls(detector, gridDim, featureLimit, borderBuffer)

	def __init__(self, detector, gridDim, featureLimit, borderBuffer=0):
		self._detector = detector

		self._gridDimensions = gridDim
		self._numCells = gridDim[0] * gridDim[1]

		self._buffer = borderBuffer

		self._featureLimit = featureLimit
		self._featuresPerCell = int(featureLimit / self._numCells)

		self._detector.featureLimit = self._featuresPerCell

	def detect(self, img, mask):
		heightStep = int((img.shape[0] - 2 * self._buffer) / self._gridDimensions[0])
		widthStep = int((img.shape[1] - 2 * self._buffer) / self._gridDimensions[1])

		# Declare features array
		features = None
		
		for i in np.arange(self._buffer, img.shape[0]-heightStep-self._buffer+1, heightStep):
			for j in np.arange(self._buffer, img.shape[1]-widthStep-self._buffer+1, widthStep):
				#print(i, i+heightStep, j, j+widthStep)
				searchMask = mask[i:(i+heightStep), j:(j+widthStep)]
				subImg = img[i:(i+heightStep), j:(j+widthStep)]

				newFeatures = self._detector.detect(subImg, searchMask)

				if (newFeatures is not None):
					newFeatures += np.array([j, i])

					if (features is None):
						features = newFeatures
					else:
						features = np.concatenate((features, newFeatures))

		return np.float32(features).reshape(-1, 2)

	def setGrid(self, gridDim):
		self._gridDimensions = gridDim
		self._numCells = gridDim[0] * gridDim[1]

		self._featuresPerCell = int(self._featureLimit / self._numCells)
		self._detector.featureLimit = self._featuresPerCell

	@property
	def featureLimit(self):
		return self._featureLimit

	@featureLimit.setter
	def featureLimit(self, limit):
		self._featureLimit = limit
		self._featuresPerCell = int(limit / self._numCells)

		self._detector.featureLimit = self._featuresPerCell


# Todo: Priority grid detector which allocates features to cells according to some heatmap