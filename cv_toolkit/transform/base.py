from abc import ABCMeta, abstractmethod

class Transform(metaclass=ABCMeta):

	@abstractmethod
	def transformPoints(self, points):
		pass

	@abstractmethod
	def transformTrack(self, track):
		pass

	@abstractmethod
	def transformImage(self, img):
		pass

