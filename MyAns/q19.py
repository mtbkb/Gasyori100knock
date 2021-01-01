import cv2
import base as bs
import numpy as np


def LoG_filter(img, K_size=5, sigma=3):
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
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = (x ** 2 + y ** 2 - 2 * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    
    K /= (2 * np.pi + sigma ** 6)
    K /= K.sum()

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


img = cv2.imread("imori_noise.jpg")
img_gray = bs.conv_to_gray(img)
q_img = LoG_filter(img_gray)
bs.show_img(q_img)