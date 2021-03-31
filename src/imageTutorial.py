
from cv2 import cv2
import sys

# ord() - integer representing unicode char

img = cv2.imread(cv2.samples.findFile("images/faceSample.jpeg"))

if img is None:
    sys.exit("Could not read the image.")
cv2.imshow("Display window", img)
k = cv2.waitKey(7000)
if k == ord("s"):
    cv2.imwrite("images/faceSample2.jpeg", img)