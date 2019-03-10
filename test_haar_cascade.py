import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)

hand_cascade = cv2.CascadeClassifier('haar_cascade/Hand_haar_cascade.xml')

while(True):
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (5+(x+w), 5+(y+h)), (0, 255,0), 2)


    cv2.imshow('Frame', frame)  
    # cv2.imshow('gray', gray)
    # cv2.imshow('thresh', thresh)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cap.destroyAllWindows()