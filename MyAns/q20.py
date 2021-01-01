import cv2
import base as bs
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("imori.jpg")

plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()