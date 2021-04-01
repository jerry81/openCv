import numpy as np
import cv2 as cv

e1 = cv.getTickCount() # number of clock cycles after ref event - clock cycle is a single electronic pulse of a CPU 
# your code execution
img1 = cv.imread('./images/faceSample.jpeg')
for i in range(5,49,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount() 
time = (e2 - e1)/ cv.getTickFrequency() # clock cycles per second - so clcok cycles/freq = time taken
print( time )
cv.imshow("blurEffect", img1)
cv.waitKey(7000)