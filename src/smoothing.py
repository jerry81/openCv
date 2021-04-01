import cv2 as cv
import numpy as np

img = cv.imread('./images/faceSample.jpeg')

kernel = np.ones((5,5),np.float32)/25 # 5x5 matrix with all ones, focal point is center, takes sum of all items with 5x5 radius (multipled by 1) then divided by 25 (number of points)

dst = cv.filter2D(img,-1,kernel) # applies kernel to every point in img 
cv.imshow('before', img)
cv.imshow('after', dst)
cv.waitKey(20000)
cv2.destroyAllWindows()