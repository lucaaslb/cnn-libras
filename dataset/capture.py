import cv2
import time
import numpy as np
import os

#img size
image_x, image_y = 64, 64

#keys 
ESC = 27 
CAPTURE = 32
dir_img_training = './pre-processed/training/'
dir_img_test = './pre-processed/test/'

QTD_TRAIN = 600
QTD_TEST = 250

def create_folder(folder_name):
    if not os.path.exists(dir_img_training + folder_name):
        os.mkdir(dir_img_training + folder_name)
    if not os.path.exists(dir_img_test + folder_name):
        os.mkdir(dir_img_test + folder_name)
    
               
def capture_images(letra, nome):
    create_folder(str(letra))
    
    cam = cv2.VideoCapture(0)

    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    folder = ''
    
    while True:

        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)

        img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

        result = img[102:298, 427:623]              

        cv2.putText(frame, folder +": "+str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("frame", frame)
        cv2.imshow("result", result)
    

        if cv2.waitKey(1) == CAPTURE:

            if t_counter <= QTD_TRAIN:
                img_name = dir_img_training + str(letra) + "/"+nome+"{}.png".format(training_set_image_name)
                save_img = cv2.resize(result, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print("{} written!".format(img_name))
                training_set_image_name += 1
                img_counter = training_set_image_name
                folder = "TRAIN"  
            
            if t_counter > QTD_TRAIN and t_counter <= (QTD_TRAIN+QTD_TEST):
                img_name = dir_img_test + str(letra) + "/"+nome+"{}.png".format(test_set_image_name)
                save_img = cv2.resize(result, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print("{} written!".format(img_name))
                test_set_image_name += 1
                img_counter = test_set_image_name
                folder = "TEST"  
                                   
            t_counter += 1

            
            if t_counter > (QTD_TRAIN+QTD_TEST):
                print('[INFO] FIM')
                break
                

        if cv2.waitKey(1) == ESC:
            break

            
      
    cam.release()
    cv2.destroyAllWindows()
    
letra = input("LETRA: ")
nome = input("NOME: ")
capture_images(letra, nome)