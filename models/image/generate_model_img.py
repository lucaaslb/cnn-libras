'''
@author Lucas Lacerda

Gerar imagem e sumario do modelo treinado

@param local e nome do arquivo de model
@param nome para salvar a imagem

'''

from keras.utils import to_categorical, plot_model 
from keras.models import load_model
import sys

model = sys.argv[1]
model = load_model(model)

model_name = sys.argv[2]

plot_model(model, to_file=model_name+'.png', show_shapes = True)

print('[INFO] Summary: ')

model.summary()