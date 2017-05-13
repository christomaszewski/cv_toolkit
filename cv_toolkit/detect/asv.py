import cv2
import numpy as np

from .base import Detector

class BoatDetector(Detector):
	""" Detector for boat with Blue painted front plate and Red painted back plate

		Todo: produce track object 

		Todo: learn thresholds from labelled dataset or load from calib file
	"""

	def __init__(self):
		self._lowerBlue = np.array([113,140,40], np.uint8)
		self._upperBlue = np.array([123,255,255], np.uint8)
		self._lowerRed = np.array([165,120,50], np.uint8)
		self._upperRed = np.array([180,255,255], np.uint8)
		self._lowerRed2 = np.array([0,50,50], np.uint8)
		self._upperRed2 = np.array([8,255,255], np.uint8)
		self._midpoint  = (0,0)

		self._featureLimit = 1

	def detect(self, img, mask):
		# Convert image to HSV space
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# Find image regions which fall into blue and red ranges
		blueMask = cv2.inRange(hsv, self._lowerBlue, self._upperBlue)
		redMask = cv2.inRange(hsv, self._lowerRed, self._upperRed)
		redMask2 = cv2.inRange(hsv, self._lowerRed2, self._upperRed2)
		redMask = cv2.addWeighted(redMask, 1.0, redMask2, 1.0, 0.0)

		# Dilate and erode mask to remove spurious detections/close gaps
		blueMask = cv2.dilate(blueMask, None, iterations=1)
		blueMask = cv2.erode(blueMask, None, iterations=1)
		# cv2.namedWindow("blue", cv2.WINDOW_NORMAL)
		# cv2.imshow("blue", blueMask)
		# Find blue contours
		blueContours = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		
		# If no blue contours find, terminate (Boat not in frame)
		if (len(blueContours) < 1):
			print("Blue Contour Not Found")
			return False

		# Choose largest blue contour
		maxBlue = max(blueContours, key=cv2.contourArea)

		# Store bounding box around blue panel
		box = cv2.minAreaRect(maxBlue)
		box = cv2.boxPoints(box)
		self._bluePanelRect = np.array(box, dtype='int')

		# Find centroid of largest blue contour
		blueMoments = cv2.moments(maxBlue)
		self._blueCentroid = np.asarray([int(blueMoments["m10"] / blueMoments["m00"]), 
										int(blueMoments["m01"] / blueMoments["m00"])])
		
		# Repeat above process for red contours
		redMask = cv2.dilate(redMask, None, iterations=1)
		redMask = cv2.erode(redMask, None, iterations=1)
		# cv2.namedWindow("red", cv2.WINDOW_NORMAL)
		# cv2.imshow("red", redMask)
		# Find red contours and choose largest
		redContours = cv2.findContours(redMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

		# If no red contours find, terminate (Boat not in frame)
		if (len(redContours) < 1):
			print("Red Contour Not Found")
			return False

		# Choose largest red contour
		maxRed = max(redContours, key=cv2.contourArea)

		# Store bounding box around red panel
		box = cv2.minAreaRect(maxRed)
		box = cv2.boxPoints(box)
		self._redPanelRect = np.array(box, dtype='int')

		# Find centroid of largest red contour
		redMoments = cv2.moments(maxRed)
		self._redCentroid = np.asarray([int(redMoments["m10"] / redMoments["m00"]), 
										int(redMoments["m01"] / redMoments["m00"])])
		diff = self._redCentroid - self._blueCentroid
		dist = np.linalg.norm(diff)
		#np.sqrt((redCentroid[0]-blueCentroid[0])**2+(redCentroid[1]-blueCentroid[1])**2)
		print("Distance between centroids: ", dist)

		rise = diff[0]
		run = diff[1]

		self._midpoint = (int(self._blueCentroid[0]+rise*0.5), int(self._blueCentroid[1]+run*0.5))

		return True

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

	@property
	def featureLimit(self):
		return self._featureLimit


	@featureLimit.setter
	def featureLimit(self, limit):
		self._featureLimit = limit




