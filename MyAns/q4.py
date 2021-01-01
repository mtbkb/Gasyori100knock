import cv2
import base as bs
import numpy as np


def binarize(img, thr):
    ret = np.zeros(img.shape, dtype='uint8')
    ret[img >= thr] = 255
    return ret


def conv_to_gray(img):

    return (0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]).astype(np.uint8)


def Otsu_binarize(img):
    img = img.astype(np.float)
    thres_set = range(255)

    argmin_thr = 0
    max_sigma_b = None
    for thr in thres_set:
        c_1 = np.array(img[img <= thr])
        c_2 = np.array(img[img > thr])
        w_1 = len(c_1)
        w_2 = len(c_2)
        if(len(c_1) == 0 or len(c_2) == 0):
            continue
        else:
            tmp_sigma_b = w_1 * w_2 * np.power(np.mean(c_1) - np.mean(c_2), 2) / np.power((w_1 + w_2), 2)
            if(max_sigma_b is None or max_sigma_b < tmp_sigma_b):
                argmin_thr = thr
                max_sigma_b = tmp_sigma_b
        
    return binarize(img.astype(np.uint8), argmin_thr), argmin_thr


img = cv2.imread("imori.jpg").astype(np.float)
img_gray = conv_to_gray(img)
img_bin, thr = Otsu_binarize(img_gray)
print("thr:", thr)
bs.show_img(img_bin)
