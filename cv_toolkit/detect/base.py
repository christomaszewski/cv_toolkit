from abc import ABCMeta, abstractmethod

class Detector(metaclass=ABCMeta):

	@abstractmethod
	def detect(self, img, mask):
		# Must return detections in some standard format
		pass

	@property
	@abstractmethod
	def featureLimit(self):
		pass

	@featureLimit.setter
	@abstractmethod
	def featureLimit(self, limit):
		pass