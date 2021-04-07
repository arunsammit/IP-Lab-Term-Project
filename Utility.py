import cv2 as cv
import os
import sys
def getImages(imgsPath, masksPath):
    images = []
    names = []
    if((not os.path.isdir(imgsPath)) or (not os.path.isdir(masksPath))):
        print("paths given as arguments are not paths to valid directory")
        print("Exiting...")
        sys.exit(1)
    for imgName in os.listdir(imgsPath):
        img = cv.imread(imgsPath + '/' + imgName)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mask = cv.imread(masksPath + '/' + imgName)
        mask[mask<128] = 0
        mask[mask>=128] = 255
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        images.append((img, mask))
        names.append(imgName)
    return images, names


