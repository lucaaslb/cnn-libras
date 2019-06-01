"""
author: Lucas Lacerda @lucaaslb

Exemplo para preditar uma imagem com o modelo treinado, exemplo de imagens no diretorio './images'

Executar:

python3 app.py 'local_imagem'

*deve ser informado o path absoluto da imagem

Exemplo:

python3 app.py images/img.png

""" 

import cv2
import numpy as np
import sys 
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image 

image_x, image_y = 64, 64


classifier = load_model('../../models/cnn_model_LIBRAS_20190530_0204.h5')
classes = 21
letras = {'0' : 'A', '1' : 'B', '2' : 'C', '3': 'D', '4': 'E', '5':'F', '6':'G', '7': 'G', '8':'I', '9':'L', '10':'M', '11': 'N', '12':'O', '13':'P', '14':'Q', '15':'R', '16':'S', '17':'T', '18':'U', '19':'V','20':'W', '21':'Y'}

def predictor(img):
            
       test_image = image.img_to_array(img)       
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       maior, class_index = -1, -1

       for x in range(classes):      
           
           if result[0][x] > maior:
              maior = result[0][x]
              class_index = x
       
       return [result, letras[str(class_index)]]

def main() :    
       
       path_img = str(sys.argv[1])              
       img = cv2.imread(path_img)       
       #img = Image.open(path_img).convert('L')
       img = cv2.resize(img, (image_x, image_y))       
       # imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
       predict = predictor(img)
       
       print('\n\n===========================\n')
       print('Imagem: ', path_img)
       print('Vetor de resultado: ', predict[0])
       print('Classe: ', predict[1])       
       print('\n===========================\n')
      
__init__ = main()