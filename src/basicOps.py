import numpy as np # for fast array calculations
import cv2 as cv
img = cv.imread('./images/faceSample.jpeg')
px = img[100][100] # equivalent to img[100, 100]
img[:,:,2] = 0
cv.imshow('noRed', img)
cv.waitKey(7000)