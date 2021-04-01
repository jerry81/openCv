import cv2 as cv
import numpy as np

img = cv.imread('./images/faceSample.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # converts to greyscale
ret,thresh1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
cv.imshow('grey', thresh1)
cv.waitKey(7000)

