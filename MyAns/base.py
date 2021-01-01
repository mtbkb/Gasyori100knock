import cv2
import numpy as np


def show_img(img):
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def conv_to_gray(img):

    return (0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]).astype(np.uint8)