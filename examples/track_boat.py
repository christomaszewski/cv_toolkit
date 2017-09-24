import cv2
import numpy as np
import os
import sys

from context import cv_toolkit

from cv_toolkit.data import Dataset
from cv_toolkit.transform.camera import UndistortionTransform
from cv_toolkit.detect.asv import BoatDetector

from primitives.track import Track


refPt = []

def select_boat(event, x, y, flags, param):
	global refPt

	if event == cv2.EVENT_LBUTTONDOWN:
		refPt=[(x,y)]
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x,y))

		cv2.rectangle(transImg, refPt[0], refPt[1], (0,255,0), 2)
		cv2.imshow("image", transImg)



cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#cv2.setMouseCallback("image", select_boat)

datasetFile = os.path.abspath(sys.argv[1])

data = Dataset.from_file(datasetFile)
undistortTransform = UndistortionTransform(data.camera)

img, timestamp = data.read()
transImg = undistortTransform.transformImage(img)

boat = cv2.selectROI("image", transImg)
frontPanel = cv2.selectROI("image", transImg)
rearPanel = cv2.selectROI("image", transImg)

# boat left
# (1357, 1517, 119, 63) (1390, 1528, 45, 35) (1422, 1526, 57, 54)

# boat center
# (1852, 1178, 96, 100) (1874, 1211, 37, 43) (1889, 1174, 52, 54)

# boat center 2
# (2100, 1378, 100, 100) (2124, 1411, 41, 43) (2150, 1382, 43, 52)

# boat right
#boat = (2369, 1580, 111, 80)
#frontPanel = (2399, 1606, 40, 39)
#rearPanel = (2434, 1586, 42, 50)
print(boat, frontPanel, rearPanel)
b = []
f = []
r = []

boatTracker = cv2.MultiTracker_create()
ok = boatTracker.add(cv2.TrackerKCF_create(), transImg, boat)
ok = boatTracker.add(cv2.TrackerKCF_create(), transImg, frontPanel)
ok = boatTracker.add(cv2.TrackerKCF_create(), transImg, rearPanel)

boatPosition = (boat[0]+boat[2]/2, boat[1]+boat[3]/2)
frontPanelPosition = (frontPanel[0]+frontPanel[2]/2, frontPanel[1]+frontPanel[3]/2)
rearPanelPosition = (rearPanel[0]+rearPanel[2]/2, rearPanel[1]+rearPanel[3]/2)

b.append(np.array(boatPosition))
f.append(np.array(frontPanelPosition))
r.append(np.array(rearPanelPosition))

db = [b, f, r]
c1 = f[-1]
c2 = r[-1]
m = (c1+c2)/2
boatTrack = Track.from_point(m, timestamp)

while(data.more()):
	img, timestamp = data.read()
	transImg = undistortTransform.transformImage(img)
	ok, boxes = boatTracker.update(transImg)

	for i, box in enumerate(boxes):
		if ok:
			position = (box[0]+box[2]/2, box[1]+box[3]/2)
			db[i].append(np.array(position))
			p1 = (int(box[0]), int(box[1]))
			p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
			color = [0.,0.,0.]
			color[i] = 200.
			cv2.rectangle(transImg, p1, p2, tuple(color), 2)
		else:
			print("Not ok")

	if ok:
		c1 = db[1][-1]
		c2 = db[2][-1]
		m = (c1+c2)/2
		coord = (int(m[0]), int(m[1]))
		cv2.circle(transImg, coord, 5, (0,0,0), -1)
		boatTrack.addObservation(m, timestamp)

	cv2.imshow("image", transImg)
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break


origTrack = undistortTransform.inverseTransformTrack(boatTrack)
origTrack.save(f"{data.name}_boat_track.yaml")

"""
while True:
	cv2.imshow("image", transImg)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("c"):
		break

if len(refPt) < 2:
	exit()

boatTracker = cv2.TrackerMIL_create()


print(refPt)
"""