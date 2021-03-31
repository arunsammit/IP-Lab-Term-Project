#%%
import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
import scipy.stats
#%%
class ClassicalModel:
    def __init__(self, images):
        self.developModel(images)
    def developModel(self, images):
        self.u = np.zeros((1, 2))
        self.var = np.zeros((1,2))
        numPixels = np.zeros((1,2))
        for img,mask in images:
            cv.imshow("image",img)
            cv.imshow("mask",mask)
            for i in range(2):
                select = mask == 255*i
                self.u[0,i]+=np.sum(img[select])
                numPixels[0,i]+=np.sum(select)
        self.u[0,0]/= numPixels[0,0]
        self.u[0,1]/= numPixels[0,1]
        for img,mask in images:
            img2 = []
            for i in range(2):
                select = mask == 255*i
                img2 = (img-self.u[0,i])**2
                self.var[0,i] += np.sum(img2[select])
        self.var[0,0]/=numPixels[0,0]
        self.var[0,1]/=numPixels[0,1]
    def logProb(self,image,i):
        return (image - self.u[0,i])**2/self.var[0,i]+np.log(self.var[0,i])
    def segmentImage(self,image):
        outputImage = np.zeros((image.shape),dtype=np.uint8)
        logProb0 = self.logProb(image, 0)
        logProb1 = self.logProb(image, 1)
        outputImage[logProb0 > logProb1 ] = 255
        return outputImage
    def segmentImages(self,images):
        for image in images:
            outputImage = self.segmentImage(image)
            cv.imshow('segmentedOutputImage',outputImage)
            cv.waitKey(0)
    def plotModel(self):
        x = np.arange(0,256,1)
        y1 = scipy.stats.norm(self.u[0,0],self.var[0,0]**.5)
        y2 = scipy.stats.norm(self.u[0,1], self.var[0, 1]**.5)
        fig, ax = plt.subplots()
        ax.plot(x,y1.pdf(x))
        ax.plot(x,y2.pdf(x))
        plt.show()

#%%
def getImages(imgsPath, masksPath):
    images = []
    for imgName in os.listdir(imgsPath):
        img = cv.imread(imgsPath +'/'+ imgName)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask = cv.imread(masksPath + '/'+ imgName)
        mask = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
        images.append((img,mask))
    return images

#%%
images = getImages('./input_image', './mask')
fp = ClassicalModel(images)
fp.plotModel()
fp.segmentImages([image[0] for image in images])
