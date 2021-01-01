import cv2
import base as bs


def changeRB(img):
    red = img[:, :, 2].copy()
    img[:, :, 2] = img[:, :, 0]
    img[:, :, 0] = red
    return img


img = cv2.imread("imori.jpg")
img = changeRB(img)
bs.show_img(img)
