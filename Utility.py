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


