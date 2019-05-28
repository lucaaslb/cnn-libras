'''
@author Lucas Lacerda 
@date 05/2019
'''
from keras.utils import to_categorical, plot_model 
from keras.optimizers import SGD
from keras import backend
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from cnn import Convolucao

import datetime
import h5py
import time

EPOCHS = 25
CLASS = 3

date = ('{date:%Y%m%d %H:%M:%S}').format(date=datetime.datetime.now())
print('[INFO] [INICIO]: ' + date + '\n')

print('[INFO] Download dataset usando keras.preprocessing.image.ImageDataGenerator')

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training',
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        shuffle=True,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(64, 64),
        color_mode = 'rgb',
        batch_size=32,
        class_mode='categorical')

# inicializar e otimizar modelo
print("[INFO] Inicializando e otimizando a CNN...")
start = time.time()

model = Convolucao.build(64, 64, 3, CLASS)
model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
              metrics=["accuracy"])

# treinar a CNN
print("[INFO] Treinando a CNN...")
classifier = model.fit_generator(
        training_set,
        steps_per_epoch=1000,
        epochs=EPOCHS,
        validation_data = test_set,
        validation_steps=400,
        shuffle = True,
        verbose=2
      )

print("[INFO] Salvando modelo treinado ...")
file_date = ('_{date:%Y%m%d_%H%M%S}').format(date=datetime.datetime.now())

model.save('models/trained_classifier_model'+file_date+'.h5')
print('[INFO] modelo: models/trained_classifier_model'+file_date+'.h5 salvo!')

end = time.time()

print("[INFO] Tempo de execução da CNN: %.2f s" %(end - start))

print("[INFO] Avaliando a CNN...")
print('[INFO] Summary: ')
model.summary()

score = model.evaluate_generator(generator=test_set, steps=100, verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

print("[INFO] Sumarizando loss e accuracy para os datasets 'train' e 'test'")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCHS), classifier.history["loss"], label="train_loss")
plt.plot(np.arange(0,EPOCHS), classifier.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,EPOCHS), classifier.history["acc"], label="train_acc")
plt.plot(np.arange(0,EPOCHS), classifier.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('models/graphics/trained_classifier_model_LIBRAS'+file_date+'.png', bbox_inches='tight')

print('[INFO] Gerando imagem do modelo de camadas da CNN')
plot_model(model, to_file='models/image/trained_classifier_model'+file_date+'.png')

date = ('{date:%Y%m%d %H:%M:%S}').format(date=datetime.datetime.now())
print('\n[INFO] [FIM]: ' + date)
print('\n\n')