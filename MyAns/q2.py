import cv2
import base as bs
import numpy as np


def conv_to_gray(img):

    return (0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]).astype(np.uint8)


img = cv2.imread("imori.jpg").astype(np.float)
img_gray = conv_to_gray(img)
bs.show_img(img_gray)
