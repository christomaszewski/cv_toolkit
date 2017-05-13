import cv2
import numpy as np

from cams import FisheyeCamera
from detect import BoatDetector

pts = []


def click_handler(event, x, y, flags, param):
	global pts

	if event is cv2.EVENT_LBUTTONDOWN:
		pts.append([x, y])
		cv2.circle(unwarped, (x, y), 5, (0, 0, 255), -1)
		cv2.imshow('image', unwarped)
	elif event is cv2.EVENT_LBUTTONUP:
		return

if __name__ == '__main__':
	img = cv2.imread('img.jpg')
	print(img.shape)
	fishCam = FisheyeCamera.from_file('cam1.yaml')
	unwarped = fishCam.undistortImage(img, cropped=False)

	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image', unwarped)

	cv2.setMouseCallback('image', click_handler)

	bd = BoatDetector()

	boatPresent = bd.detect(unwarped, None)

	if boatPresent:
		cv2.drawContours(unwarped, [bd._bluePanelRect, bd._redPanelRect], -1, (0, 255, 0), 1)
		pts = bd._redPanelRect
	else:
		cv2.waitKey(3000)
		exit() 

	cv2.imshow('image', unwarped)
	cv2.waitKey(30000)

	#pts = [[969, 1518], [1010, 1470], [1049, 1506], [1005, 1557]]

	while(not boatPresent or len(pts) < 4):
		cv2.waitKey(100)

	img_size = unwarped.shape[:-1]
	print(img_size)
	src_pts = np.array(pts) 
	dst_pts = np.array([[0,0], [40,0], [40,20], [0,20]])
	dst_pts_offset = np.array([[100,100], [140,100], [140,120], [100,120]])

	h1, status = cv2.findHomography(src_pts, dst_pts)
	h2, status = cv2.findHomography(src_pts, dst_pts_offset)
	print(src_pts)
	print(h1)
	print(h2)

	r3 = np.cross(h1[:,0], h1[:,1])

	h3 = np.copy(h1)
	h3[:,2] = r3

	#h3 = h3 / h3[2,2]


	thetaX = np.arctan(h3[2,1]/h3[2,2]) 
	#np.arctan2(h3[2,1], h3[2,2])
	thetaY = -np.arcsin(h3[2,0])
	#np.arctan2(-h3[2,0], np.sqrt(h3[2,1]**2 + h3[2,2]**2))
	thetaZ = np.arctan(h3[1,0]/h3[0,0])
	#np.arctan2(h3[1,0], h3[0,0])

	print(thetaX, thetaY, thetaZ)

	Rx = np.array([[1,0,0], [0, np.cos(thetaX), -np.sin(thetaX)], [0, np.sin(thetaX), np.cos(thetaX)]])
	Ry = np.array([[np.cos(thetaY), 0, -np.sin(thetaY)], [0, 1, 0], [np.sin(thetaY), 0, np.cos(thetaY)]])
	Rz = np.array([[np.cos(thetaZ), np.sin(thetaZ), 0], [-np.sin(thetaZ), np.cos(thetaZ), 0], [0,0,1]])
	print(Rx)


	RTotal = np.dot(Rz, Ry)
	RTotal = np.dot(RTotal, Rx)

	print(h3)
	print(RTotal)


	origin = np.array([[0],[0],[1]])
	new_origin = np.dot(h3, origin)
	new_origin2 = np.dot(RTotal, origin)
	print(origin)
	print(new_origin)
	print(new_origin2)

	h3[:,2] = np.array([[0, 0, 1]])

	new_img = cv2.warpPerspective(unwarped, h2, (240, 220))

	cv2.imshow('image', new_img)
	cv2.waitKey(0)

	cv2.imwrite('m2.jpg', new_img)


