import numpy as np
import copy

class FrameTransformation(object):
	""" Class that can transform between frames (typically image and world)

	"""

	def __init__(self, imgSize, cameraModel):
		self._camModel = cameraModel
		self._imgHeight, self._imgWidth = imgSize
		print(self._imgWidth, self._imgHeight)

		corners = [[0,0], [0, self._imgHeight], [self._imgWidth, 0], [self._imgWidth, self._imgHeight]]

		pts = self._camModel.undistortPoints(corners)

		#todo: change to crop to smallest rectangle enclosing all corners
		x, y = pts[0]
		r, b = pts[3]
		self._cropBounds = [int(x), int(y), int(r), int(b)]
		print(self._cropBounds)

	def transformImg(self, img):
		undistortedImg = self._camModel.undistortImage(img)

		croppedImg = undistortedImg[self._cropBounds[1]:self._cropBounds[3], 
					self._cropBounds[0]:self._cropBounds[2]]

		return croppedImg

	def transformTrackForPlotting(self, track):
		""" Does everything transform track does except for the coordinate flip
			so tracks can be easily plotted on image

		"""
		newTrack = copy.deepcopy(track)

		ptSeq = newTrack.getPointSequence()

		undistortedPtSeq = self._camModel.undistortPoints(ptSeq)
		undistortedPtSeq = np.reshape(undistortedPtSeq, (-1,2))

		# Adjust for cropping
		xOffset, yOffset, _, _ = self._cropBounds
		undistortedPtSeq[:, 0] -= xOffset
		undistortedPtSeq[:, 1] -= yOffset

		newTrack.updatePointSequence(undistortedPtSeq)

		return newTrack

	def transformTrack(self, track):
		""" Applies unwarping, crop, and coordinate flip to put origin in lower
			left hand corner. Returns a new copy of the track.
		"""

		newTrack = copy.deepcopy(track)

		ptSeq = newTrack.getPointSequence()

		undistortedPtSeq = self._camModel.undistortPoints(ptSeq)
		undistortedPtSeq = np.reshape(undistortedPtSeq, (-1,2))

		# Adjust for cropping
		xOffset, yOffset, _, _ = self._cropBounds
		undistortedPtSeq[:, 0] -= xOffset
		undistortedPtSeq[:, 1] -= yOffset


		# Flip y coordinate to shift origin down to left bottom corner
		newImgHeight = self._cropBounds[3] - self._cropBounds[1]
		undistortedPtSeq[:, 1] *= -1.0
		undistortedPtSeq[:, 1] += newImgHeight

		newTrack.updatePointSequence(undistortedPtSeq)

		return newTrack

	def getCroppedExtents(self):
		return [(0, self._cropBounds[2]-self._cropBounds[0]), 
				(0, self._cropBounds[3]-self._cropBounds[1])]

