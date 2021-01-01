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


def rgb2hsv(_img):
    img = _img.astype(np.float) / 255
    img_max = np.max(img, axis=2).copy()
    img_min = np.min(img, axis=2).copy()
    min_arg = np.argmin(img, axis=2)
    hsv = np.zeros(img.shape, dtype='float')

    ind = np.where(min_arg == 0)
    print(ind)
    hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (img_max[ind] - img_min[ind]) + 60
    ind = np.where(min_arg == 2)
    hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (img_max[ind] - img_min[ind]) + 180
    ind = np.where(min_arg == 1)
    hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (img_max[ind] - img_min[ind]) + 300
    hsv[..., 0][np.where(img_max == img_min)] = 0
    
    hsv[..., 1] = img_max - img_min

    hsv[..., 2] = img_max
    print(img.shape)
    print(img_max.shape)
    print(img_max)
    return hsv


def HSV2BGR(_img, hsv):
    img = _img.copy() / 255.

    # get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()

    out = np.zeros_like(img)

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    C = S
    H_ = H / 60.
    X = C * (1 - np.abs( H_ % 2 - 1))
    Z = np.zeros_like(H)

    vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i+1)))
        out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
        out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
        out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

    out[np.where(max_v == min_v)] = 0
    out = np.clip(out, 0, 1)
    out = (out * 255).astype(np.uint8)

    return out


img = cv2.imread("imori.jpg").astype(np.float)
#hsv = BGR2HSV(img)
hsv = rgb2hsv(img)
bs.show_img(hsv)

out = HSV2BGR(img, hsv)
bs.show_img(out)