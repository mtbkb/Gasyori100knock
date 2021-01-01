import cv2
import base as bs
import numpy as np


def binarize(img):
    ret = np.zeros(img.shape, dtype='uint8')
    ret[img >= 128] = 255
    return ret


def conv_to_gray(img):

    return (0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]).astype(np.uint8)


img = cv2.imread("imori.jpg").astype(np.float)
img_gray = conv_to_gray(img)
img_bin = binarize(img_gray)
print(img_bin)
bs.show_img(img_bin)
