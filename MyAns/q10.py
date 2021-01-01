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


def median_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    
    ##padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    tmp = out.copy()
    for y in range(H):
        for x in range(W):
            for c_i in range(C):
                out[pad + y, pad + x, c_i] = np.median(tmp[y: y + K_size, x: x + K_size, c_i])
    
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out


def pool2d(_img, w_size=8):
    out = img.copy()

    h, w, c = _img.shape
    nh = int(h / w_size)
    nw = int(w / w_size)
    for y in range(nh):
        for x in range(nw):
            for c_i in range(c):
                out[w_size * y:w_size * (y + 1), w_size * x: w_size * (x + 1), c_i] = np.max(_img[w_size * y:w_size * (y + 1), w_size * x: w_size * (x + 1), c_i])
    
    return out


img = cv2.imread("imori_noise.jpg")
q_img = median_filter(img)
bs.show_img(q_img)
