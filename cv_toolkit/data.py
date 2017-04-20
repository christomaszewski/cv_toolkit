import yaml
import os
import glob
import cv2

from queue import Queue
from threading import Thread

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
	def from_file(cls, file):
		with open(file, mode='r') as f:
			return yaml.load(f)

	def __init__(self, location=None, start=0, end=0, cam=None):
		# Location of dataset on disk
		self._location = location

		# First frame of dataset
		self._startFrame = start

		# Last frame of dataset
		self._endFrame = end

		# Path to camera calibration file for dataset
		self._camera = cam

		# Init frame list to []
		self._frames = []

		# Init frame index to dataset start frame
		self._currFrameIndex = start

		# Init buffer frame index to start frame
		self._bufferFrameIndex = start

		# Set image buffering to stopped
		self._bufferThreadStopped = True

		# Init image buffer
		self._imgBuffer = Queue(maxsize=128)

		# Setup buffering thread
		self._bufferingThread = Thread(target=self._bufferUpdate, args=())
		self._bufferingThread.daemon = True

	def save(self, filename):
		with open(filename, 'w') as f:
			yaml.dump(self, f)

	def more(self):
		return self._currFrameIndex < self._endFrame

	def read(self):
		print("reading image")
		self._currFrameIndex += 1
		return self._imgBuffer.get()

	def load(self):
		""" Load bulky components of dataset

			Loads image file names, start image buffer, etc
		"""

		# Check if dataset location exists
		if (not os.path.exists(self._location)):
			print("Error: Dataset location does not exist")
			return

		self._frames = sorted(glob.glob(self._location + '/frame*.jpg'))

		# Check if startFrame is beyond dataset bounds
		if (self._startFrame > len(self._frames)):
			print("Error: Start frame is beyond dataset bounds")
			return

		# Check if endFrame is beyond dataset bounds
		if (self._endFrame > len(self._frames)):
			print("Warning: End frame beyond dataset bounds, setting to end of dataset")

		# Start buffering threads
		self._bufferThreadStopped = False
		self._bufferingThread.start()


	def _bufferUpdate(self):
		while True:
			if (self._bufferThreadStopped):
				return

			if (not self._imgBuffer.full()):
				# Load next image into buffer
				print("loading img into buffer")
				img = cv2.imread(self._frames[self._bufferFrameIndex])
				self._imgBuffer.put(img)

				if (self._bufferFrameIndex >= self._endFrame):
					# Reached end of dataset, stop buffering
					self._bufferThreadStopped = True
				else:
					self._bufferFrameIndex += 1
			else:
				print("buffer full")

	def construct(self, location, cam, start=0, end=None):
		print("Constructing")
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


	def test(self):
		print(self._location)
		print(self._startFrame)
		print(self._endFrame)
		print(self._camera)

	@classmethod
	def to_yaml(cls, dumper, data):
		""" Serializes dataset parameters to yaml for output to file
			
			Does not output bulky components such as image filenames
			or image buffer
		"""
		dict_rep = {'location':data._location, 'startFrame':data._startFrame,
					'endFrame':data._endFrame, 'camera':data._camera}

		print(dict_rep)

		node = dumper.represent_mapping(cls.yaml_tag, dict_rep)
		return node


	@classmethod
	def from_yaml(cls, loader, node):
		dict_rep = loader.construct_mapping(node)
		print("Loading from yaml")
		print(dict_rep)
		print(dict_rep['location'])
		return cls(dict_rep['location'], dict_rep['startFrame'], 
			dict_rep['endFrame'], dict_rep['camera'])

