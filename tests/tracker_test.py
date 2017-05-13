import cv2
import numpy as np

from data import Dataset
from detect import BoatDetector
from cams import FisheyeCamera


datasetFilename = "../../../datasets/llobregat_boat.yaml"

data = Dataset.from_file(datasetFilename, unwarp=True)

cv2.namedWindow('img', cv2.WINDOW_NORMAL)

boatD = BoatDetector()

boatDetected = False

img = []

while (not boatDetected and data.more()):
	img = data.read()

	boatDetected = boatD.detect(img, None)
	cv2.imshow('img', img)
	cv2.waitKey(10000)




# Build box around both panels
print(boatD._bluePanelRect, boatD._redPanelRect)
panels = np.vstack((boatD._redPanelRect, boatD._bluePanelRect))

box = cv2.minAreaRect(panels)
box = cv2.boxPoints(box)

x,y = np.min(box, axis=0)
w,h = np.max(box, axis=0)
w -= x
h -= y

box = (x, y, w, h)

boatTracker = cv2.Tracker_create("MIL")

boatTracker.init(img, box)
print("Tracker initialized")

while (data.more()):
	img = data.read()
	boatDetected, box = boatTracker.update(img)

	if boatDetected:
		p1 = (int(box[0]), int(box[1]))
		p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
		cv2.rectangle(img, p1, p2, (0,0,255), 2)

	cv2.imshow('img', img)
	k = cv2.waitKey(1) & 0xff
	if k == 27 : break

