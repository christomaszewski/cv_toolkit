import cv2
import numpy as np

from .base import Detector

from primitives.keypoint import KeyPoint


class GridDetector(Detector):

	"""
	@classmethod
	def using_shi_tomasi(cls, gridDim, featureLimit, borderBuffer=0):
		detector = ShiTomasiDetector()
		return cls(detector, gridDim, featureLimit, borderBuffer)"""

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
		features = []
		detect_method = self._detector.detect
		buff = self._buffer
		loopcount = 0
		for i in np.arange(buff, img.shape[0]-heightStep-buff+1, heightStep):
			for j in np.arange(buff, img.shape[1]-widthStep-buff+1, widthStep):
				#print(i, i+heightStep, j, j+widthStep)
				searchMask = mask[i:(i+heightStep), j:(j+widthStep)]
				subImg = img[i:(i+heightStep), j:(j+widthStep)]

				newFeatures = detect_method(subImg, searchMask)

				# Offset feature locations according to grid cell
				if newFeatures is not None:
					newFeatures = newFeatures.reshape(-1, 2)
					newFeatures += np.array([[j,i]])

					features.extend(list(newFeatures))

				loopcount += 1


		print("Grid Detector loops:", loopcount)
		return np.asarray(features)

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


class KeyPointGridDetector(GridDetector):
	"""
	def __init__(self, detector, gridDim, featureLimit, borderBuffer=0):
		self._detector = detector

		self._gridDimensions = gridDim
		self._numCells = gridDim[0] * gridDim[1]

		self._buffer = borderBuffer

		self._featureLimit = featureLimit
		self._featuresPerCell = int(featureLimit / self._numCells)

		self._detector.featureLimit = self._featuresPerCell
		"""

	def detect(self, img, mask):
		heightStep = int((img.shape[0] - 2 * self._buffer) / self._gridDimensions[0])
		widthStep = int((img.shape[1] - 2 * self._buffer) / self._gridDimensions[1])

		# Declare features array
		features = []
		detect_method = self._detector.detect
		buff = self._buffer
		loopcount = 0
		for i in np.arange(buff, img.shape[0]-heightStep-buff+1, heightStep):
			for j in np.arange(buff, img.shape[1]-widthStep-buff+1, widthStep):
				#print(i, i+heightStep, j, j+widthStep)
				searchMask = mask[i:(i+heightStep), j:(j+widthStep)]
				subImg = img[i:(i+heightStep), j:(j+widthStep)]

				newFeatures = detect_method(subImg, searchMask)

				# Offset feature locations according to grid cell
				if newFeatures is not None:
					newFeatures = newFeatures.reshape(-1, 2)
					newFeatures += np.array([[j,i]])

					features.extend(list(newFeatures))

				loopcount += 1


		print("Grid Detector loops:", loopcount)
		return np.asarray(features)


	def detectKeyPoints(self, img, mask):
		heightStep = int((img.shape[0] - 2 * self._buffer) / self._gridDimensions[0])
		widthStep = int((img.shape[1] - 2 * self._buffer) / self._gridDimensions[1])

		# Declare features array
		keyPoints = []
		descriptors = []
		angles = []
		detect_method = self._detector.detectKeyPoints
		buff = self._buffer
		loopcount = 0
		for i in np.arange(buff, img.shape[0]-heightStep-buff+1, heightStep):
			for j in np.arange(buff, img.shape[1]-widthStep-buff+1, widthStep):
				#print(i, i+heightStep, j, j+widthStep)
				searchMask = mask[i:(i+heightStep), j:(j+widthStep)]
				subImg = img[i:(i+heightStep), j:(j+widthStep)]

				newKP, newDescriptors = detect_method(subImg, searchMask)

				if newKP is not None and len(newKP) > 0:
					newKeyPoints = []
					offset = np.asarray([[j,i]])
					for kp in newKP:
						KP = KeyPoint.from_cv_keypoint(kp)
						KP.offset(offset)
						newKeyPoints.append(KP)

					keyPoints.extend(newKeyPoints)
					descriptors.extend(newDescriptors)

				loopcount += 1


		print("Grid Detector loops:", loopcount)
		return (keyPoints, np.asarray(descriptors))

	def compute(self, img, keyPoints):
		return self._detector.compute(img, keyPoints)

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