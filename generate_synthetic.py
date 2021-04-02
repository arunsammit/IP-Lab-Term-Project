import cv2 as cv
import os
import numpy as np


img = np.zeros((512, 512))
mask = np.zeros((512, 512))

x1 = y1 = 128
x2 = y2 = 128+256 # thresh
sigma_back = 10
sigma_fore = 15

mask[x1:x2+1, y1:y2+1] = 255

ht = mask.shape[0]
wd = mask.shape[1]

for id, (mu_back, mu_fore) in enumerate([(80,120), (130,140), (60,70)]):
    for i in range(ht):
        for j in range(wd):
            if mask[i][j] == 0:
                # back
                img[i][j] = np.random.normal(mu_back, sigma_back)
            else:
                # fore
                img[i][j] = np.random.normal(mu_fore, sigma_fore)
    cv.imwrite("input_synthetic/image_" + str(id) + ".jpg", img)
    cv.imwrite("mask_synthetic/image_" + str(id) + ".jpg", mask)

