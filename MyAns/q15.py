import cv2
import base as bs
import numpy as np


def prewitt_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    
    ##padding
    pad = K_size // 2
    out_v = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out_v[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    out_h = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out_h[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ##kernel
    K_h = np.zeros((K_size, K_size), dtype=np.float)
    K_v = np.zeros((K_size, K_size), dtype=np.float)
    K_h[:, 0] = -1
    K_h[:, 2] = 1
    K_v[0, :] = -1
    K_v[2, :] = 1

    tmp = out_v.copy()
    for y in range(H):
        for x in range(W):
            for c_i in range(C):
                out_v[pad + y, pad + x, c_i] = np.sum(K_v * tmp[y: y + K_size, x: x + K_size, c_i])
                out_h[pad + y, pad + x, c_i] = np.sum(K_h * tmp[y: y + K_size, x: x + K_size, c_i])
    
    out_v = np.clip(out_v, 0, 255)
    out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
    out_h = np.clip(out_h, 0, 255)
    out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out_v, out_h


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
img_gray = bs.conv_to_gray(img)
q_img_v, q_img_h = prewitt_filter(img_gray)
bs.show_img(q_img_v)
bs.show_img(q_img_h)
