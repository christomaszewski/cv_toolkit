import cv2
import numpy as np

class GridFeatureDetector(object):
	""" Class to run feature detector on an image in order to generate uniform
		number of measurements in each grid cell across image. Used to deal with
		feature rich regions hoarding entire feature quota

		Currently runs very slow - needs better structure/optimization

	"""

	def __init__(self, detector, gridDim=(2,2), partitionMethod='subimage', borderBuffer=0):
		# only supports cv2.goodFeaturesToTrack right now
		# partitionMethod subimage is fastest method
		# buffer describes region around image border to ignore
		# todo: add support for ORB
		self._featureDetector = detector

		partitionFuncName = '_' + partitionMethod
		self._partitionFunc = getattr(self, partitionFuncName, self._nopartition)

		self._buffer = borderBuffer

		self.setGrid(gridDim)

	def setGrid(self, gridDim):
		self._gridDimensions = gridDim
		self._numCells = gridDim[0] * gridDim[1]

	def detect(self, img, mask, params, numFeatures=None):
		if ('maxCorners' in params):
			# goodFeaturesToTrack parameter
			self._numTotalFeatures = params['maxCorners']
		elif (numFeatures is not None):
			self._numTotalFeatures = numFeatures
		else:
			# Default number of features
			self._numTotalFeatures = 1000

		self._numFeaturesPerCell = int(self._numTotalFeatures / self._numCells)

		heightStep = int((img.shape[0] - 2 * self._buffer) / self._gridDimensions[0])
		widthStep = int((img.shape[1] - 2 * self._buffer) / self._gridDimensions[1])

		paramCopy = params.copy()

		features = self._partitionFunc(img, mask, paramCopy, heightStep, widthStep)

		return features

	# Partition Functions
	def _mask(self, img, mask, params, heightStep, widthStep):
		""" Runs Feature detection on subsets of image using mask. 

			Note: 
				This method is substantially slower than the subimage method
		"""

		# Set desired number of features to features per cell
		params['maxCorners'] = self._numFeaturesPerCell

		# Declare features array
		features = None

		# Define full mask
		searchMask = np.zeros_like(mask)

		for i in np.arange(self._buffer, img.shape[0]-heightStep-self._buffer+1, heightStep):
			for j in np.arange(self._buffer, img.shape[1]-widthStep-self._buffer+1, widthStep):
				searchMask[i:(i+heightStep), j:(j+widthStep)] = mask[i:(i+heightStep), j:(j+widthStep)]

				newFeatures = self._featureDetector(img, mask=searchMask, **params)

				# Reset search mask 
				searchMask[i:(i+heightStep), j:(j+widthStep)] = 0

				if (features is None):
					features = newFeatures
				elif (newFeatures is not None):
					features = np.concatenate((features, newFeatures))

		return features

	
	def _subimage(self, img, mask, params, heightStep, widthStep):
		# Set desired number of features to features per cell
		params['maxCorners'] = self._numFeaturesPerCell

		# Declare features array
		features = None
		
		for i in np.arange(self._buffer, img.shape[0]-heightStep-self._buffer+1, heightStep):
			for j in np.arange(self._buffer, img.shape[1]-widthStep-self._buffer+1, widthStep):
				#print(i, i+heightStep, j, j+widthStep)
				searchMask = mask[i:(i+heightStep), j:(j+widthStep)]
				subImg = img[i:(i+heightStep), j:(j+widthStep)]

				newFeatures = self._featureDetector(subImg, mask=searchMask, **params)

				if (newFeatures is not None):
					newFeatures += np.array([j, i])

					if (features is None):
						features = newFeatures
					else:
						features = np.concatenate((features, newFeatures))

		return features

	def _nopartition(self, img, mask, params, heightStep, widthStep):
		params['maxCorners'] = self._numTotalFeatures

		features = self._featureDetector(img, mask=mask, **params)

		return features


class BoatDetector(object):
	""" Detector for boat with Blue painted front plate and Red painted back plate

		Todo: produce track object 

		Todo: learn thresholds from labelled dataset or load from calib file
	"""

	def __init__(self):
		self._lowerBlue = np.array([95,50,50], np.uint8)
		self._upperBlue = np.array([115,255,255], np.uint8)
		self._lowerRed = np.array([165,50,50], np.uint8)
		self._upperRed = np.array([180,255,255], np.uint8)
		self._lowerRed2 = np.array([0,50,50], np.uint8)
		self._upperRed2 = np.array([10,255,255], np.uint8)
		self._midpoint  = (0,0)

	def detect(self, img):
		# Convert image to HSV space
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Find image regions which fall into blue and red ranges
		blueMask = cv2.inRange(hsv, self._lowerBlue, self._upperBlue)
		redMask = cv2.inRange(hsv, self._lowerRed, self._upperRed)
		redMask2 = cv2.inRange(hsv, self._lowerRed2, self._upperRed2)
		redMask = cv2.addWeighted(redMask, 1.0, redMask2, 1.0, 0.0)

		# Erode and dilate mask to remove spurious detections
		blueMask = cv2.erode(blueMask, None, iterations=1)
		blueMask = cv2.dilate(blueMask, None, iterations=1)
		#cv2.namedWindow("blue", cv2.WINDOW_NORMAL)
		#cv2.imshow("blue", blueMask)
		# Find blue contours
		blueContours = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		
		# If no blue contours find, terminate (Boat not in frame)
		if (len(blueContours) < 1):
			print("Blue Contour Not Found")
			return False

		# Choose largest blue contour
		maxBlue = max(blueContours, key=cv2.contourArea)

		# Find centroid of largest blue contour
		blueMoments = cv2.moments(maxBlue)
		self._blueCentroid = np.asarray([int(blueMoments["m10"] / blueMoments["m00"]), 
										int(blueMoments["m01"] / blueMoments["m00"])])
		
		# Repeat above process for red contours
		redMask = cv2.erode(redMask, None, iterations=1)
		redMask = cv2.dilate(redMask, None, iterations=1)
		#cv2.namedWindow("red", cv2.WINDOW_NORMAL)
		#cv2.imshow("red", redMask)
		# Find red contours and choose largest
		redContours = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

		# If no red contours find, terminate (Boat not in frame)
		if (len(redContours) < 1):
			print("Red Contour Not Found")
			return False

		# Choose largest red contour
		maxRed = max(redContours, key=cv2.contourArea)

		# Find centroid of largest red contour
		redMoments = cv2.moments(maxRed)
		self._redCentroid = np.asarray([int(redMoments["m10"] / redMoments["m00"]), 
										int(redMoments["m01"] / redMoments["m00"])])
		diff = self._redCentroid - self._blueCentroid
		dist = np.linalg.norm(diff)
		#np.sqrt((redCentroid[0]-blueCentroid[0])**2+(redCentroid[1]-blueCentroid[1])**2)
		print("Distance between centroids: ", dist)


		# If distance between centroids of detected panels matches boat report detection
		if (dist >= 6 and dist <= 13):
			cv2.drawContours(img,maxBlue,-1,(0,0,255),7)
			cv2.drawContours(img,maxRed,-1,(255,0,0),7)

			cv2.circle(img, tuple(self._redCentroid), 5, (0,0,0), thickness=-1, lineType=8, shift=0)
			cv2.circle(img, tuple(self._blueCentroid), 5, (0,0,0), thickness=-1, lineType=8, shift=0)
			cv2.line(img, tuple(self._redCentroid), tuple(self._blueCentroid), (0,0,0), thickness=2)

			rise = diff[0]
			run = diff[1]

			self._midpoint = (int(self._blueCentroid[0]+rise*0.5), int(self._blueCentroid[1]+run*0.5))
			cv2.circle(img, self._midpoint, 5, (0,0,0), thickness=-1)
			print("Midpoint: ", self._midpoint)
			return True

		return False

	def getROI(self, img):
		# Convert image to HSV space
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Find image regions which fall into blue and red ranges
		blueMask = cv2.inRange(hsv, self._lowerBlue, self._upperBlue)
		redMask = cv2.inRange(hsv, self._lowerRed, self._upperRed)
		redMask2 = cv2.inRange(hsv, self._lowerRed2, self._upperRed2)
		redMask = cv2.addWeighted(redMask, 1.0, redMask2, 1.0, 0.0)

		boatMask = cv2.addWeighted(redMask, 1.0, blueMask, 1.0, 0.0)

		boatMask = cv2.erode(boatMask, None, iterations=1)
		boatMask = cv2.dilate(boatMask, None, iterations=2)
		cv2.imshow("blue", boatMask)


		boatContours = cv2.findContours(boatMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		maxBoat = max(boatContours, key=cv2.contourArea)

		return maxBoat

	@property
	def midpoint(self):
		return self._midpoint