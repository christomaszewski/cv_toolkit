import cv2
import dill
import glob
import numpy as np

class FisheyeCameraModel(object):
	""" Class for loading calibration settings and rectifying images accordingly

	"""

	def __init__(self, camIntrinsics=None, distortionCoeff=None, imgSize=None):
		self._K = camIntrinsics
		self._D = distortionCoeff

		self._newK = camIntrinsics

		self._cropBounds = [0,0,0,0]

		self._imgSize = imgSize
		self._initialized = False

	def loadModel(self, fileName):
		with open(fileName, 'rb') as f:
			(self._K, self._D) = dill.load(f)

	def saveModel(self, fileName):
		with open(fileName, 'wb') as f:
			dill.dump((self._K, self._D), f)

	def initialize(self, imgSize):
		# Todo: store this as part of the model
		self._imgSize = imgSize

		# Compute new K
		newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self._K,
			self._D, self._imgSize, np.eye(3), balance=1.0, new_size=self._imgSize,
			fov_scale=1.75)

		# Reset to center of image for now
		newK[0,2] = self._imgSize[0] / 2
		newK[1,2] = self._imgSize[1] / 2

		self._newK = newK

		self._initialized = True

	def undistortPoints(self, points):
		points = np.asarray(points)
		assert points.ndim == 2 and points.shape[1] == 2

		if points.ndim == 2:
			points = np.expand_dims(points, 0)

		undistorted = cv2.fisheye.undistortPoints(points.astype(np.float32), self._K, self._D, R=np.eye(3), P=self._newK)

		return np.squeeze(undistorted)

	def undistortImage(self, img):
		map1, map2 = cv2.fisheye.initUndistortRectifyMap(self._K, self._D, np.eye(3),
			self._newK, self._imgSize, cv2.CV_16SC2)

		undistorted = cv2.remap(img, map1, map2,
			interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

		return undistorted

	@property
	def initialized(self):
		return self._initialized


class FisheyeCalibration(object):
	""" Class to produce a FisheyeCameraModel from a set of calibration images

		Only standard checkerboard images supported for now
	"""

	def __init__(self, checkerboardDim=(9,6)):
		self.__checkerboardDim = checkerboardDim

		self.__termCrit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.00001)

		# Prepare object points corresponding to calibration checkerboard
		x, y = checkerboardDim
		self.__objPoints = np.zeros((1, x * y, 3), np.float32)
		self.__objPoints[0,:,:2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
		
		# Allocate space for camera intrinsics and distortion coefficients
		self.__K = np.zeros((3,3))
		self.__D = np.zeros((4,1))

	def computeModel(self, imgDir, fileExtension="JPG"):
		# Note Windows path...
		imageFiles = glob.glob(imgDir + "\\*." + fileExtension)

		objPoints = [] # 3d points in real world space
		imgPoints = [] # 2d points in image plane

		for fileName in imageFiles:
			img = cv2.imread(fileName)
			grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

			ret, corners = cv2.findChessboardCorners(grayImg, self.__checkerboardDim,
				cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

			if (ret):
				print("Found corners in image", fileName)
				objPoints.append(self.__objPoints)

				# todo: check if window size needs to adjust based on checkerboard
				corners = cv2.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), self.__termCrit)
				imgPoints.append(corners.reshape(1, -1, 2))
			else:
				print("Could not find corners in image", fileName)

		numImgPoints = len(imgPoints)
		rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(numImgPoints)]
		tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(numImgPoints)]

		print("Finished processing images, computing model")
		# 512 should correspond to missing cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT = 1 << 9
		err, _, _, _, _ = cv2.fisheye.calibrate(objPoints, imgPoints, grayImg.shape[::-1],
			self.__K, self.__D, rvecs, tvecs,
			cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW+512, 
			self.__termCrit)


		fisheyeCam = FisheyeCameraModel(self.__K, self.__D)

		return fisheyeCam