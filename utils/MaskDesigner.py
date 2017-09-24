import cv2
import numpy as np

drawing = False
size = 10
imgShow = []

def click_handler(event, x, y, flags, param):
	global ix, iy, drawing, size, imgShow

	if event is cv2.EVENT_LBUTTONDOWN:
		drawing = True
	elif event is cv2.EVENT_LBUTTONUP:
		drawing = False

	if drawing:
		cv2.circle(mask, (x, y), size, 0, -1)
		cv2.circle(img, (x, y), size, (0, 0, 255), -1)
		imgShow = np.copy(img)
	else:
		imgShow = np.copy(img)
		cv2.circle(imgShow, (x,y), size, (255, 255, 255), -1)


img = cv2.imread('frame_boat.jpg')
mask = np.load('mask.npy')
print(mask)

img[:,:,2] = np.max([img[:,:,2], mask], axis=0)
np.max
imgShow = np.copy(img)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)

cv2.setMouseCallback('image', click_handler)

while(1):
	#imgShow = np.copy(img)
	cv2.imshow('image', imgShow)
	cv2.imshow('mask', mask)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('='):
		size += 5
	elif k == ord('-'):
		size -= 5
	elif k == ord('s'):
		np.save('mask.npy', mask)
	elif k == 27:
		break

#np.save('mask.npy', mask[:,:,2])

cv2.destroyAllWindows()