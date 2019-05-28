import cv2
import time
import numpy as np
import os

#img size
image_x, image_y = 100, 100

#keys 
ESC = 27 
CAPTURE = 32
dir_img_training = './dataset/training/'
dir_img_test = './dataset/test/'
autor = 'lb'

def nothing(x):
    pass

def create_folder(folder_name):
    if not os.path.exists(dir_img_training + folder_name):
        os.mkdir(dir_img_training + folder_name)
    if not os.path.exists(dir_img_test + folder_name):
        os.mkdir(dir_img_test + folder_name)
    
               
def capture_images(ges_name):
    create_folder(str(ges_name))
    
    cam = cv2.VideoCapture(0)

    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    listImage = [1,2,3,4,5]

    for loop in listImage:
        while True:

            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

            imcrop = img[102:298, 427:623]
          
            result = cv2.GaussianBlur(imcrop,(5,5),0)

            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            cv2.imshow("frame", frame)
            cv2.imshow("result", result)
       

            if cv2.waitKey(1) == CAPTURE:

                if t_counter <= 350:
                    img_name = dir_img_training + str(ges_name) + "/"+autor+"{}.png".format(training_set_image_name)
                    save_img = cv2.resize(result, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    training_set_image_name += 1
                   
                
                if t_counter > 350 and t_counter <= 470:
                    img_name = dir_img_test + str(ges_name) + "/"+autor+"{}.png".format(test_set_image_name)
                    save_img = cv2.resize(result, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    test_set_image_name += 1
                    
                                 
                t_counter += 1
                img_counter = t_counter
                
                if t_counter > 470:
                    print('[INFO] fim')
                    break

            if cv2.waitKey(1) == ESC:
                break

              
      
    cam.release()
    cv2.destroyAllWindows()
    
ges_name = input("LETRA: ")
capture_images(ges_name)