import cv2 as cv
import os
def getImages(imgsPath, masksPath):
    images = []
    names = []
    for imgName in os.listdir(imgsPath):
        img = cv.imread(imgsPath + '/' + imgName)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask = cv.imread(masksPath + '/' + imgName)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        images.append((img, mask))
        names.append(imgName)
    return images, names

def segmentImages(model, images):
    outputImages = []
    for image in images:
        outputImage = model.segmentImage(image)
        cv.imshow("image", image)
        cv.imshow('segmentedOutputImage', outputImage)
        cv.waitKey(0)
        outputImages.append(outputImage)
    return outputImages
