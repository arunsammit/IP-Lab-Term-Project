import cv2 as cv
import os
import numpy as np


img = np.zeros((512, 512),dtype=np.uint8)
mask = np.zeros((512, 512),dtype=np.uint8)

x1 = y1 = 128
x2 = y2 = 128+256 # thresh
sigma_back = 10
sigma_fore = 15

mask[x1:x2+1, y1:y2+1] = 255

ht = mask.shape[0]
wd = mask.shape[1]

for id, (mu_back, mu_fore) in enumerate([(112,144), (154,186), (70,102)]):
    img = np.random.default_rng(seed = 99).normal(mu_back,sigma_back,size=(512,512))
    np.putmask(img, mask == 255,np.random.default_rng(
        seed=99).normal(mu_fore, sigma_fore, size=(x2-x1, y2-y1)))
    cv.imwrite("input_synthetic/image_" + str(id) + ".jpg", img)
    cv.imwrite("mask_synthetic/image_" + str(id) + ".jpg", mask)

