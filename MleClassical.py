
import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
import scipy.stats
from Utility import getImages

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


if __name__ == "__main__":
    if(len(sys.argv)!=3):
        print("usage python3 MleClassical.py input_image/ input_mask/")
        sys.exit(1)
    images, names = getImages(sys.argv[1], sys.argv[2])
    input_image_type = sys.argv[1].split('_')[1]
    fp = ClassicalModel(images)
    figs = []
    dice = []
    specificity = []
    sensitivity = []
    total_pixel = 0

    fig, ax = plt.subplots()
    fp.plotModel(ax)
    figs.append(fig)
    fig.savefig(f'./output_{input_image_type}/classical/plots/plot.jpg')

    for i, image in enumerate([image for image in images]):
        outputImage = fp.segmentImage(image[0])
        
        # print(image[0].shape)
        # print(outputImage.shape)
        dn = (outputImage == 0)
        dp = (outputImage == 255)

        tn = (image[1] == 0)
        tp = (image[1] == 255)
        
        dtp = (dp & tp)
        dtn = (dn & tn)
        # print(np.unique(outputImage))
        # print(np.unique(image[1]))
        
        size = outputImage.shape[0] * outputImage.shape[1]
        total_pixel += size

        specificity.append((np.sum(dtn) / np.sum(tn))*size)
        sensitivity.append((np.sum(dtp) / np.sum(tp))*size)
        dice.append(((2.0 * np.sum(dtp)) / (np.sum(dp) + np.sum(tp)))*size)

        # cv.imshow("image", image[0])
        # cv.imshow("mask", image[1])
        # cv.imshow('segmentedOutputImage', outputImage)
        cv.imwrite(f'output{input_image_type}/classical/'+names[i], outputImage)
        cv.waitKey(0)

    print("Specificity : ", str(np.sum(specificity)*100/total_pixel))
    print("Sensitivity : ", str(np.sum(sensitivity)*100/total_pixel))
    print("Dice Measure : ", str(np.sum(dice)*100/total_pixel))
    # plt.show()

