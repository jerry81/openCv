import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0) # accesses webcam
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True: # loop forever
    # Capture frame-by-frame
    ret, frame = cap.read() # destructuring the tuple response 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # converts to greyscale
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'): # press q to exit 
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()