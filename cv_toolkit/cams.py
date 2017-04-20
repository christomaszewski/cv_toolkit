import yaml
import cv2
import numpy as np

class FisheyeCamera(yaml.YAMLObject):
	""" Object representing a fisheye camera model

		self._K: Camera intrinsics
		self._D: Camera distortion coefficients
		self._imgSize: Camera image size in pixels

	"""
	yaml_tag = '!Fisheye_Camera'

	@classmethod
	def from_file(cls, filename):
		with open(filename, mode='r') as f:
			cam = yaml.load(f)
			cam._initialize()
			return cam

	def __init__(self, K, D, imgSize, name=None, fmt=None):
		self._K = np.asarray(K)
		self._D = np.asarray(D)
		self._imgSize = tuple(imgSize)

		self._name = name
		self._format = fmt

		self._initialize()

	def _initialize(self):
		self._K = np.asarray(self._K)
		self._D = np.asarray(self._D)
		self._imgSize = tuple(self._imgSize)

		print(self._imgSize)

		# Initialize new K matrix
		newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self._K,
			self._D, self._imgSize, np.eye(3), balance=1.0, fov_scale=0.7)

		# Reset to center of image for now
		newK[0,2] = self._imgSize[0] / 2
		newK[1,2] = self._imgSize[1] / 2		

		self._newK = newK

	def save(self, filename):
		with open(filename, mode='w') as f:
			yaml.dump(self, f)

	def undistortImage(self, img, cropped=False):
		map1, map2 = cv2.fisheye.initUndistortRectifyMap(self._K, self._D, np.eye(3),
			self._newK, self._imgSize, cv2.CV_16SC2)

		undistorted = cv2.remap(img, map1, map2,
			interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

		if (not cropped):
			return undistorted

		imgHeight, imgWidth = self._imgSize

		corners = [[0,0], [0, imgHeight], [imgWidth, 0], [imgWidth, imgHeight]]

		pts = self.undistortPoints(corners)

		x = int(min(pts[0,0], pts[1,0]))
		y = int(min(pts[0,1], pts[2,1]))
		r = int(max(pts[2,0], pts[3,0]))
		b = int(max(pts[2,1], pts[3,1]))

		print(pts)
		print(x, y, r, b)

		croppedImg = undistorted[y:b, x:r]
		return croppedImg

		#return croppedImg

	def undistortPoints(self, points):
		points = np.asarray(points)
		assert points.ndim == 2 and points.shape[1] == 2

		if points.ndim == 2:
			points = np.expand_dims(points, 0)

		undistorted = cv2.fisheye.undistortPoints(points.astype(np.float32), self._K, 
			self._D, R=np.eye(3), P=self._newK)

		return np.squeeze(undistorted)

	@classmethod
	def to_yaml(cls, dumper, data):
		""" Serializes dataset parameters to yaml for output to file
			
			Does not save newK, saves imgSize as list for cleaner yaml
		"""
		sizeNode = list(data._imgSize)

		# Convert to list to avoid numpy objects in yaml file
		kNode = data._K.tolist()
		dNode = data._D.tolist()
		newKNode = data._newK.tolist()

		dict_rep = {'_K':kNode, '_D':dNode, '_imgSize':sizeNode, 
					'_name':data._name, '_format':data._format}

		node = dumper.represent_mapping(cls.yaml_tag, dict_rep)
		return node
