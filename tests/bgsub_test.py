import cv2
import numpy as np

from data import Dataset

print(cv2.useOptimized())

datasetFilename = "../../../datasets/llobregat_boat.yaml"

data = Dataset.from_file(datasetFilename, unwarp=True)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
	img = data.read()
	fgmask = fgbg.apply(img)
	cv2.imshow('img',fgmask)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
	    break