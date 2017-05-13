import cv2
import numpy as np

from primitives.grid import Grid

from context import cv_toolkit

from cv_toolkit.data import Dataset
from cv_toolkit.detect.features import ShiTomasiDetector
from cv_toolkit.detect.adapters import GridDetector

from cv_toolkit.transform.camera import UndistortionTransform


datasetFilename = "../../../datasets/llobregat_short.yaml"
data = Dataset.from_file(datasetFilename)

img = data.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = ShiTomasiDetector()

gd = GridDetector(detector, (15, 20), 1000, 0)
transformation = UndistortionTransform(data.camera)


detections = gd.detect(gray, data.mask)
print(detections)

for point in detections:
	cv2.circle(img, tuple(point), 3, (255,0,0), -1)

undistortedImg = transformation.transformImage(img)
undistortedPoints = transformation.transformPoints(detections)

for point in undistortedPoints:
	cv2.circle(undistortedImg, tuple(point), 3, (255,0,0), -1)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)

cv2.namedWindow('undistorted', cv2.WINDOW_NORMAL)
cv2.imshow('undistorted', undistortedImg)
cv2.waitKey(0)