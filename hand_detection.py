import numpy as np
import cv2

ESC = 27 
cap_region_x_begin = 0.6  # start point/total width
cap_region_y_end = 0.8  # start point/total width
hand_cascade = cv2.CascadeClassifier('haar_cascade/hand_haar_cascade.xml')

# open web cam
camera = cv2.VideoCapture(0)    

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1) #flip the frame horizontally
  
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #cv2.CascadeClassifier.detectMultiScale(image, rejectLevels, levelWeights[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize[, outputRejectLevels]]]]]]) 
    hands = hand_cascade.detectMultiScale(imgGray, 1.5, 7)


    for (x, y, w, h) in hands:
        # cv2.rectangle(img, (x1, y1), (x2, y2), color)
        cv2.rectangle(frame, (x, y), (5+(x+w), 5+(y+h)), (0, 255,0), 2)

        # REGION OF INTEREST - ROI 
        roi = frame[y:5+(y+h), x:5+(x+w)]

        cv2.imshow('roi', roi) #region of interest
                           
    
    cv2.imshow('WEBCAM', frame) #origin
   
    if cv2.waitKey(1) == ESC:
      break

camera.release()
camera.destroyAllWindows()