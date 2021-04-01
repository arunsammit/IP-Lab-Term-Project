#%%
import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
import scipy.stats
from Utility import getImages
#%%
class ClassicalModel:
    def __init__(self, images):
        self.developModel(images)
    def developModel(self, images):
        self.u = [0.0,0.0]
        self.var = [0.0,0.0]
        numPixels = [0,0]
        for i in range(2):
            for img,mask in images:
                select = mask == 255*i
                self.u[i]+=np.sum(img[select])
                numPixels[i]+=np.sum(select)
            self.u[i]/= numPixels[i]
            
            for img,mask in images:
                select = mask == 255*i
                img2 = (img-self.u[i])**2
                self.var[i] += np.sum(img2[select])
            self.var[i]/=numPixels[i]
    def logProb(self,image,i):
        return (image - self.u[i])**2/self.var[i]+np.log(self.var[i])
    def segmentImage(self,image):
        outputImage = np.zeros((image.shape),dtype=np.uint8)
        logProb0 = self.logProb(image, 0)
        logProb1 = self.logProb(image, 1)
        outputImage[logProb0 > logProb1 ] = 255
        return outputImage
    def plotModel(self, ax):
        x = np.arange(0,256,1)
        y1 = scipy.stats.norm(self.u[0],self.var[0]**.5)
        y2 = scipy.stats.norm(self.u[1], self.var[1]**.5)
        ax.plot(x,y1.pdf(x),label='background')
        ax.plot(x,y2.pdf(x),label='foreground')

#%%
if __name__ == "__main__":
    images,names = getImages('./input_image', './mask')
    fp = ClassicalModel(images)
    for i,image in enumerate([image[0] for image in images]):
        outputImage = fp.segmentImage(image)
        cv.imshow("image", image)
        cv.imshow('segmentedOutputImage', outputImage)
        cv.waitKey(0)
        cv.imwrite('output_image/classical/'+names[i],outputImage)
    fig, ax = plt.subplots()
    fp.plotModel(ax)
    ax.legend()
    plt.show()
    fig.savefig('output_image/classical/plots/model.jpg')
