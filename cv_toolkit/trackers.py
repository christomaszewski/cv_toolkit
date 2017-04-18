from abc import ABCMeta, abstractmethod

import numpy as np
import cv2

from ..core import tracking
from . import detectors
class Tracker(metaclass=ABCMeta):

	@abstractmethod
	def processImage(self, img, timestamp):
		pass

	@abstractmethod
	def getTracks(self):
		pass


class LKOpticalFlowTracker(Tracker):

	def __init__(self, lkParams, featureParams, detectionInterval=0.1):
		self._lkParams = lkParams
		self._featureParams = featureParams

		self._prevImg = None
		self._prevTimestamp = None

		self._activeTracks = []
		self._detectionInterval = detectionInterval
		self._prevDetectionTime = None

		self._deviationThreshold = 1

	def processImage(self, img, timestamp):
		# Todo: Check if input is already grayscale
		grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		trackEndpoints = self.getTrackEndpoints()

		newTracks = []

		gridDetector = detectors.GridFeatureDetector(cv2.goodFeaturesToTrack, (15,20), borderBuffer=35)

		# If features have never been detected or detectionInverval has lapsed
		if (self._prevDetectionTime is None or timestamp - self._prevDetectionTime > self._detectionInterval):
			#print("Finding New Features")
			self._prevDetectionTime = timestamp
			searchMask = np.zeros_like(grayImg)
			searchMask[:] = 255

			# Masks right side of image
			#searchMask[:, -320:] = 0

			# Masks shadow region
			
			searchMask[:300, -1200:] = 0
			searchMask[300:850, -1100:] = 0
			searchMask[:, -300:] = 0
			searchMask[850:1100, -800:-400] = 0
			

			# Mask all the current track end points
			for (x,y) in trackEndpoints:
				cv2.circle(searchMask, (x,y), 5, 0, -1)

			p = gridDetector.detect(grayImg, searchMask, self._featureParams)

			if (p is not None):
				for x, y in np.float32(p).reshape(-1, 2):
					newTracks.append(tracking.Track((x,y), timestamp))

		if (self._prevImg is not None):
			prevPoints = np.float32(trackEndpoints).reshape(-1,1,2)

			# Run LK forwards
			nextPoints, status, error = cv2.calcOpticalFlowPyrLK(self._prevImg,
				grayImg, prevPoints, None, **self._lkParams)

			# Run LK in reverse
			prevPointsRev, status, error = cv2.calcOpticalFlowPyrLK(grayImg,
				self._prevImg, nextPoints, None, **self._lkParams)

			# Compute deviation between original points and back propagation
			dev = abs(prevPoints-prevPointsRev).reshape(-1,2)

			# For each pair of points, take max deviation in either axis
			maxDev = dev.max(-1)

			# Check against max deviation threshold allowed
			matchQuality = maxDev < self._deviationThreshold

			for (track, point, match) in zip(self._activeTracks, nextPoints.reshape(-1,2), matchQuality):
				if (match):
					track.addObservation(point, timestamp)
					newTracks.append(track)

				# Todo: Handle tracks that have been lost


		self._activeTracks = newTracks

		self._prevImg = grayImg
		self._prevTimestamp = timestamp


	def getTrackEndpoints(self):
		endpoints = []

		for track in self._activeTracks:
			(time, position) = track.getLastObservation()
			endpoints.append(position)

		return endpoints

	def getTracks(self):
		return self._activeTracks