import numpy as np
import matplotlib.pyplot as plt
import cv2

from context import cv_toolkit

from cv_toolkit.data import Dataset
from viz_toolkit.view import ImageView

plt.ion()

datasetFilename = "../../../datasets/llobregat_boat.yaml"

data = Dataset.from_file(datasetFilename)

img = data.read()

imgView = ImageView(img)

imgView.plot(pause=10)
