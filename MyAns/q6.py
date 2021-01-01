import cv2
import base as bs
import numpy as np


def quant4(_img):
    ret = np.zeros_like(_img, dtype='uint8')
    
    print(_img.shape)
    for i in range(3):
        ind = _img[..., i] < 64
        print(ind.shape)
        ret[..., i][ind] = 32
        ind = _img[..., i] < 128
        i_inserted = ret[..., i] == 0
        ret[..., i][np.logical_and(ind, i_inserted)] = 96
        ind = _img[..., i] < 192
        i_inserted = ret[..., i] == 0
        ret[..., i][np.logical_and(ind, i_inserted)] = 160
        i_inserted = ret[..., i] == 0
        ret[..., i][i_inserted] = 224
    
    return ret
        

img = cv2.imread("imori.jpg").astype(np.float)
q_img = quant4(img)
bs.show_img(q_img)
