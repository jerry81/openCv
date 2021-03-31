import numpy as np
import cv2 as cv
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
# cv.line(img,(0,0),(511,511),(255,0,0),5) # canvas, start x and y, end x and y, color, thickness?

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
print('pts before', pts)
pts = pts.reshape((-1,1,2))
print('after reshape', pts)
# reshape took the array elements and wrapped them in another array
cv.polylines(img,[pts],True,(0,255,255))

cv.imshow('drawTest', img)
cv.waitKey(7000)