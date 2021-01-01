import cv2
import base as bs
import numpy as np


def motion_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    
    ##padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ##kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for i in range(3):
        K[i, i] = 1 / 3

    tmp = out.copy()
    for y in range(H):
        for x in range(W):
            for c_i in range(C):
                out[pad + y, pad + x, c_i] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c_i])
    
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


img = cv2.imread("imori.jpg")
q_img = motion_filter(img)
bs.show_img(q_img)
