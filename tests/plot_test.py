import numpy as np
import matplotlib.pyplot as plt
import cv2

from data import Dataset
from plot import ImageView

plt.ion()

datasetFilename = "../../../datasets/llobregat_boat.yaml"

data = Dataset.from_file(datasetFilename, unwarp=True)

img = data.read()

imgView = ImageView(img)

imgView.plot(pause=100)
