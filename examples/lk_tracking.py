import cv2
import numpy as np

from primitives.grid import Grid

from context import cv_toolkit

from cv_toolkit.data import Dataset
from cv_toolkit.detect.features import ShiTomasiDetector
from cv_toolkit.detect.adapters import GridDetector

from cv_toolkit.track.flow import LKOpticalFlowTracker

from cv_toolkit.transform.camera import UndistortionTransform

def mainFunc():
	datasetFilename = "../../../datasets/llobregat_short.yaml"
	data = Dataset.from_file(datasetFilename)

	img = data.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	winSize = (15,15)
	maxLevel = 0
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

	lk = LKOpticalFlowTracker(winSize, maxLevel, criteria, gray)

	detector = ShiTomasiDetector()

	gd = GridDetector(detector, (15, 20), 1000, 0)
	transformation = UndistortionTransform(data.camera)

	points = gd.detect(gray, data.mask)
	oldPoints = points
	cv2.namedWindow('img', cv2.WINDOW_NORMAL)

	loopcount = 0

	while(data.more()):
		img = data.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		undistortedImg = transformation.transformImage(img)

		newPoints = lk.trackPoints(points, gray)

		points = np.asarray([p for p in newPoints if p is not None])

		if len(points) > 100:
			uPoints = transformation.transformPoints(points)
			for p in uPoints:
				cv2.circle(undistortedImg, tuple(p), 3, (255, 0, 0), -1)
		else:
			points = gd.detect(gray, data.mask)

		cv2.imshow('img', undistortedImg)
		if (cv2.waitKey(1) & 0xFF) == 27:
			break

		loopcount += 1
		print(loopcount)

	cv2.destroyAllWindows()

if __name__ == '__main__':
	mainFunc()