import yaml
import os
import glob
import cv2
import os
import numpy as np

from threading import Thread

from .cams import FisheyeCamera

class Dataset(yaml.YAMLObject):
	""" Object representing a dataset

		self._location: Location of dataset on disk
		self._startFrame: First frame of dataset
		self._endFrame: Last frame of dataset
		self._currFrameIndex: Current fram that will be read
		self._bufferFrameIndex: Next frame to load into buffer

	"""
	yaml_tag = '!Image_Dataset'

	@classmethod
	def from_file(cls, filename, autoload=True):
		with open(filename, mode='r') as f:
			data = yaml.load(f)
			data.filename = filename
			data.load()

			return data

	def __init__(self, name, date, location, folder, start=0, end=0, fps=30, cam=None):
		# Human Readable Dataset name
		self._name = name

		# Date dataset was gathered
		self._date = date

		# Location dataset was gathered
		self._location = location

		# Last known dataset path and filename
		self._filename = None
		self._path = None

		# Relative path of images on disk
		self._imgFolder = folder

		# First frame of dataset
		self._startFrame = start

		# Last frame of dataset
		self._endFrame = end

		# FPS of dataset
		self._fps = fps
		self._timestep = 1.0/fps

		# Relative Path to camera calibration file for dataset
		self._cameraFile = cam
		self._camera = None

		# Image mask as numpy array (for masking background etc)
		self._mask = None

		# Init frame list to []
		self._frames = []

		# Init frame index to dataset start frame
		self._currFrameIndex = start

		# Buffered current image
		self._currFrame = None

	def save(self, filename=None):
		if (filename is None):
			filename = self._name + '.yaml'

		self._filename = filename

		with open(filename, 'w') as f:
			yaml.dump(self, f)

	def more(self):
		return self._currFrameIndex < self._endFrame

	def read(self):
		self._currFrame = cv2.imread(self._frames[self._currFrameIndex])
		timestamp = self.currentTime
		self._currFrameIndex += 1

		return (self._currFrame, timestamp)

	def load(self):
		""" Load bulky components of dataset

			Loads image file names, start image buffer, etc
		"""

		# Check if dataset location exists
		if (not os.path.exists(self._path + self._imgFolder)):
			print("Error: Dataset location does not exist")
			return

		# Check if mask file present in image folder
		if (os.path.exists(self._path + self._imgFolder + '/mask.npy')):
			self._mask = np.load(self._path + self._imgFolder + '/mask.npy')

		self._frames = sorted(glob.glob(self._path + self._imgFolder + '/frame*.jpg'))

		# Check if camera file is available
		if (self._cameraFile is not None and os.path.exists(self._path + self._cameraFile)):
			# Todo: Handle all camera types or don't load camera object
			self._camera = FisheyeCamera.from_file(self._path + self._cameraFile)
		else:
			print("Error: Cannot load dataset camera calibration")

		# Check if startFrame is beyond dataset bounds
		if (self._startFrame > len(self._frames)):
			print("Error: Start frame is beyond dataset bounds")
			return

		# Check if endFrame is beyond dataset bounds
		if (self._endFrame > len(self._frames)):
			print("Warning: End frame beyond dataset bounds, setting to end of dataset")

		# Load first image
		self._currFrame = cv2.imread(self._frames[self._currFrameIndex])

		# Initialize imageSize
		imgHeight, imgWidth = self._currFrame.shape[:2]
		self._imgSize = (imgWidth, imgHeight)


	def construct(self, location, cam, start=0, end=None):
		# Check if dataset location exists
		if (not os.path.exists(location)):
			print("Error: Dataset location does not exist")
		else:
			self._location = location

		# Get number of frames
		self._frames = sorted(glob.glob(location + '/frame*.jpg'))

		if (end is None):
			end = len(self._frames)

		if (start > end):
			print("Error: Start frame index is beyond end of dataset")
		elif (start < 0):
			print("Warning: Start frame index less than 0, setting to 0")
			start = 0

		self._startFrame = start
		self._endFrame = end

		self._camera = cam

		self._imgSize = cam.imgSize

	@classmethod
	def to_yaml(cls, dumper, data):
		""" Serializes dataset parameters to yaml for output to file
			
			Does not output bulky components such as image filenames
			or image buffer
		"""
		dict_rep = {'name':data._name, 'date':data._date, 'location':data._location,
					'imgFolder':data._imgFolder, 'startFrame':data._startFrame,
					'endFrame':data._endFrame, 'fps':data._fps, 'camera':data._cameraFile}

		print(dict_rep)

		node = dumper.represent_mapping(cls.yaml_tag, dict_rep)
		return node


	@classmethod
	def from_yaml(cls, loader, node):
		dict_rep = loader.construct_mapping(node)
		print("Loading from yaml")
		init_params = ['name', 'date', 'location', 'imgFolder', 
						'startFrame', 'endFrame', 'fps', 'camera']
		params = [dict_rep[x] for x in init_params]
		return cls(*params)


	@property
	def filename(self):
		return self._filename

	@filename.setter
	def filename(self, newFilename):
		self._path = os.path.dirname(newFilename) + '/'
		self._filename = newFilename

	@property
	def imgSize(self):
		return self._imgSize

	@property
	def currentTime(self):
		return self._currFrameIndex * self._timestep

	@property
	def mask(self):
		return self._mask

	@property
	def camera(self):
		return self._camera

	@property
	def name(self):
		return self._name