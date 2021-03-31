#%%
import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
import scipy.stats
#%%
class ProposedModel:
    def __init__(self,images):
        self.buildModel(images)
    def buildModel(self,images):
        self.u = [np.zeros((2,1)), np.zeros(2,1)]
        self.var = [np.zeros(2,2), np.zeros((2,2))]
        numPixels = [0,0]
        for i in range(2):
            totPixelCnt = 0 
            for img,mask in images:
                select = mask == 255*i
                self.u[i] += np.sum(img[select])
                self.u[i][1, 0] +=np.sum(img) 
                numPixels[i] += np.sum(select)
                totPixelCnt += img.size
            self.u[i][0,0] /= numPixels[0]
            self.u[i][1,0] /= totPixelCnt

            for img, mask in images:
                mean_intensity_centred = np.mean(img)-self.u[i][0,1]
                select = mask == 255*i
                img_centred = img-self.u[i]
                img_centred_2 = (img_centred)**2
                self.var[i][0,0] += np.sum(img_centred_2[select])
                self.var[i][0,1] += np.sum(img_centred*mean_intensity_centred)
                self.var[i][1,1] += img.size*mean_intensity_centred 
            self.var[i][1,0] = self.var[i][0,1]
            self.var[i][0,0] /= numPixels[0]
            self.var[i][0,1] /= totPixelCnt
            self.var[i][1,0] = self.var[i][0,1]
