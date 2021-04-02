
import numpy as np
import cv2 as cv
import sys
import os
import matplotlib.pyplot as plt
import scipy.stats
from Utility import getImages


class ProposedModel:
    def __init__(self, images):
        self.buildModel(images)

    def buildModel(self, images):
        self.u = [np.zeros((2, 1)), np.zeros((2, 1))]
        self.var = [np.zeros((2, 2)), np.zeros((2, 2))]
        numPixels = [0, 0]
        for i in range(2):
            for img, mask in images:
                select = mask == 255*i
                self.u[i][0, 0] += np.sum(img[select])
                self.u[i][1, 0] += np.mean(img)*np.sum(select) 
                numPixels[i] += np.sum(select)
            self.u[i] /= numPixels[i]

            for img, mask in images:
                mean_intensity_centred = np.mean(img)-self.u[i][1, 0]
                select = mask == 255*i
                img_centred = img-self.u[i][0, 0]
                img_centred_2 = (img_centred)**2
                self.var[i][0, 0] += np.sum(img_centred_2[select])
                self.var[i][0, 1] += np.sum(img_centred[select]*mean_intensity_centred)
                self.var[i][1, 1] += np.sum(select)*(mean_intensity_centred**2)
            self.var[i][1, 0] = self.var[i][0, 1]
            self.var[i] /= numPixels[i]

    def logProb(self, img, i):
        mean_centred = np.mean(img)-self.u[i][1, 0]
        pixel_centred = img - self.u[i][0, 0]
        var_inv = np.linalg.inv(self.var[i])
        det = np.linalg.det(self.var[i])
        var0 = var_inv[0, 0]
        var1 = var_inv[1, 1]
        var01 = var_inv[0, 1]
        retVal = var0*(pixel_centred**2) + 2*var01*mean_centred*pixel_centred + \
            var1*mean_centred**2 + np.log(np.abs(det))
        return retVal

    def segmentImage(self, img):
        outputImage = np.zeros((img.shape), dtype=np.uint8)
        logProb0 = self.logProb(img, 0)
        logProb1 = self.logProb(img, 1)
        outputImage[logProb1 < logProb0] = 255
        return outputImage

    def plotModel(self, img, ax):
        x = np.arange(0, 256, 1)
        x = x.reshape((x.size, 1))
        y = np.ones((x.size, 1))*np.mean(img)
        ip = np.dstack((x,y))
        y1 = scipy.stats.multivariate_normal(self.u[0].flatten(), self.var[0])
        y2 = scipy.stats.multivariate_normal(self.u[1].flatten(), self.var[1])
        ax.plot(x, y1.pdf(ip), label='background')
        ax.plot(x, y2.pdf(ip), label='foreground')
        ax.legend()
        return ax


if __name__ == "__main__":
    images, names = getImages('./input_image', './mask')
    fp = ProposedModel(images)
    figs = []
    dice = []
    specificity = []
    sensitivity = []
    total_pixel = 0

    for i, image in enumerate([image for image in images]):
        outputImage = fp.segmentImage(image[0])
        fig, ax = plt.subplots()
        fp.plotModel(image[0], ax)
        figs.append(fig)
        fig.savefig(f'./output_image/proposed/plots/{names[i]}')
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
        cv.imwrite('output_image/proposed/'+names[i], outputImage)
        cv.waitKey(0)

        mask = image[1]
        mask_comp = 255 - mask
        fig1, ax1 = plt.subplots()

        histogram_fore = cv.calcHist([image[0]], [0], mask, [256], [1, 256])
        histogram_back = cv.calcHist([image[0]], [0], mask_comp, [256], [1, 256])
        
        ax1.plot(histogram_fore, label='foreground')
        ax1.plot(histogram_back, label='background')
        ax1.legend()
        fig1.savefig("output_image/proposed/histogram_"+names[i])
 
    print("Specificity : ", str(np.sum(specificity)*100/total_pixel))
    print("Sensitivity : ", str(np.sum(sensitivity)*100/total_pixel))
    print("Dice Measure : ", str(np.sum(dice)*100/total_pixel))
    # plt.show()


    

        
