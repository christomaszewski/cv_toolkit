import cv2
from skimage.feature import corner_shi_tomasi, corner_subpix, corner_peaks
from skimage.color import rgb2gray
import timeit


def get_features1(img):
	coords = corner_peaks(corner_shi_tomasi(img), min_distance=10, num_peaks=500)

def get_features2(img):
	cv2.goodFeaturesToTrack(img, maxCorners=500, qualityLevel=0.9, minDistance=10, blockSize=7)


img = cv2.imread('img.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = rgb2gray(img[:,:,::-1])

#print(timeit.repeat("get_features1(img1)", number=10, repeat=3, globals=globals()))
print(timeit.repeat("get_features2(img2)", number=10, repeat=30, globals=globals()))