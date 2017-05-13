import cv2
import numpy as np

from primitives.grid import Grid

from context import cv_toolkit

from cv_toolkit.data import Dataset
from cv_toolkit.detect.features import ORBDetector
from cv_toolkit.detect.adapters import KeyPointGridDetector

from cv_toolkit.track.features import SparseFeatureTracker

from cv_toolkit.transform.camera import UndistortionTransform

from memory_profiler import profile

@profile
def mainFunc():
	datasetFilename = "../../../datasets/llobregat_short.yaml"
	data = Dataset.from_file(datasetFilename)

	img = data.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	orbParams = dict(nfeatures=100, nlevels=1)

	detector = ORBDetector(orbParams)

	gd = KeyPointGridDetector(detector, (20, 30), 100000, 0)
	transformation = UndistortionTransform(data.camera)

	sfTracker = SparseFeatureTracker(gd)
	points = sfTracker.initialize(gray, data.mask)

	cv2.namedWindow('img', cv2.WINDOW_NORMAL)

	loopcount = 0

	while(data.more()):
		img = data.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		undistortedImg = transformation.transformImage(img)

		newPoints, indices = sfTracker.trackPoints(points, gray, data.mask)

		points = np.asarray([np.squeeze(p.point) for p in newPoints if p is not None])

		if len(points) > 10:
			uPoints = transformation.transformPoints(points)
			for p in uPoints:
				cv2.circle(undistortedImg, tuple(p), 5, (255, 0, 0), -1)
		else:
			print("Detecting new points")
			points = sfTracker.initialize(gray, data.mask)

		points = newPoints

		cv2.imshow('img', undistortedImg)
		if (cv2.waitKey(1) & 0xFF) == 27:
			break

		loopcount += 1
		print(loopcount)

	cv2.destroyAllWindows()

if __name__ == '__main__':
	mainFunc()