import numpy as np
import cv2

ESC = 27 
cap_region_x_begin = 0.6  # start point/total width
cap_region_y_end = 0.8  # start point/total width

# open web cam
camera = cv2.VideoCapture(0)    

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1) #flip the frame horizontally
  
   # REGION OF INTEREST - ROI 
   # cv2.rectangle(img, (x1, y1), (x2, y2), color)
    cv2.rectangle(frame,(int(cap_region_x_begin * frame.shape[1]),10), (frame.shape[1],int(cap_region_y_end * frame.shape[0])), (122, 10, 100))
    
   
    roi = frame[10:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] 

    imgGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    imgContours = np.zeros(roi.shape, np.uint8)
    ret, thresh = cv2.threshold(imgGray, 127,255,0)  
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        color = np.random.randint(0, 255,(3)).tolist()
        cv2.drawContours(imgContours,[cnt],0,color,2)
           

    cv2.imshow('WEBCAM', frame) #origin
    cv2.imshow('roi', roi) #region of interest
    cv2.imshow('contours', imgContours) # roi - contours 
   
   
    if cv2.waitKey(1) == ESC:
        break