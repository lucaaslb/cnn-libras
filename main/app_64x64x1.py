"""
author: Lucas Lacerda @lucaaslb

Exemplo para preditar uma imagem capturada de um video real time para um modelo treinado com o formato de imagem (64x64x1)

""" 

import cv2
import numpy as np
from keras.models import load_model
from PIL import Image 
from keras.preprocessing import image

image_x, image_y = 64,64


classifier = load_model('../models/other_models/model_epoch_48_98.6_final.h5')

classes = 21
letras = {'0' : 'A', '1' : 'B', '2' : 'C', '3': 'D', '4': 'E', '5':'F', '6':'G', '7': 'I', '8':'L', '9':'M', '10':'N', '11': 'O', '12':'P', '13':'Q', '14':'R', '15':'S', '16':'T', '17':'U', '18':'V', '19':'W','20':'Y'}


def predictor():          
       test_image = Image.open('../temp/img.png').convert('L')
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       maior, class_index = -1, -1

       for x in range(classes):      
           
           if result[0][x] > maior:
              maior = result[0][x]
              class_index = x
       
       return [result, letras[str(class_index)]]
       
    
cam = cv2.VideoCapture(0)

img_counter = 0

img_text = ['','']
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)


    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    imcrop = img[102:298, 427:623]
    
    output = np.ones((150, 150, 3)) * 255 #imagem 150x150, com fundo branco e 3 canais para as cores
        
    cv2.putText(output, str(img_text[1]), (15, 130), cv2.FONT_HERSHEY_TRIPLEX, 6, (255, 0, 0))
    
    cv2.imshow("ROI", imcrop)
    cv2.imshow("FRAME", frame)
    cv2.imshow("PREDICT", output)
   
    imggray = cv2.cvtColor(imcrop,cv2.COLOR_BGR2GRAY)
    
    img_name = "../temp/img.png"
    save_img = cv2.resize(imggray, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()
    print(str(img_text[0]))
        

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()