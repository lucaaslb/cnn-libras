import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def trainModel():
  
  import matplotlib.pyplot as plt
  import seaborn as sns
  from IPython.display import Image
  
  #CSV that contains images to train
  train = pd.read_csv('../input/sign_mnist_train.csv')

  train.head()
  print('train.shape: ', train.shape)

  #Get the 8566 row of the label column (784 columns)
  labels = train['label'].values
  print('labels: ', labels)

  #Remove as equal classes, leaving 24 classes (24 different images)
  unique_val = np.array(labels)
  print('np.unique(unique_val): ', np.unique(unique_val))

  plt.figure(figsize = (18,8))
  sns.countplot(x =labels)
  train.drop('label', axis = 1, inplace = True)

  #Train has the values of 8566 lines X 784 columns
  print('train.values: ', train.values)
  print('train.values.shape: ', train.values.shape)

  #Copy the array values
  images = train.values

  #returns to an array of 8566 rows X 784 columns
  images = np.array([i.flatten() for i in images])
  print('images: ', images)
  print('images.shape: ', images.shape)

  print('labels after Binarizer: ', labels)

  #Transforms labels into an array of classes (24 classes) per 8566 rows. The 0 does not belong to the classes, 1 belongs to the class
  label_binrizer = LabelBinarizer()
  labels = label_binrizer.fit_transform(labels)
  print('labels: ', labels)

  print('labels.shape: ', labels.shape)

  from sklearn.model_selection import train_test_split

  #Parameters to train network
  x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)
  import keras
  from keras.models import Sequential
  from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
  batch_size = 128
  #Number of classes of network
  num_classes = 24
  #Defines the epochs to train the network
  epochs = 50
  x_train = x_train / 255
  x_test = x_test / 255
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  plt.imshow(x_train[0].reshape(28,28))
  model = Sequential()
  model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
  model.add(MaxPooling2D(pool_size = (2, 2)))

  model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2, 2)))

  model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))

  model.add(MaxPooling2D(pool_size = (2, 2)))
  model.add(Flatten())
  model.add(Dense(128, activation = 'relu'))
  model.add(Dropout(0.20))
  model.add(Dense(num_classes, activation = 'softmax'))

  model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

  #model.fit train the model
  history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)

  #Saves the model
  model.save('../trained-model/mnistmodel.h5')
  
  predictTests()

def predict(img):
  
  from keras.models import load_model
  model = load_model('../trained-model/mnistmodel.h5')
  
  npImg = np.array(img)

  print('img.shape: ', npImg.shape)

  #Resizes the array to 4 dimensions
  imgToPredict = npImg.reshape(1,28,28,1)
  
  print('imgToPredict.shape: ', imgToPredict.shape)
  print(imgToPredict)
  #Save the array to a file
  #np.save('imgMatriz', lista)

  y_pred = model.predict(imgToPredict)
  print('y_pred: ', y_pred)
  return y_pred
  
def predictTests():
  
  from keras.models import load_model
  
  #CSV that contains images to test the trained model
  test = pd.read_csv('../input/sign_mnist_test.csv')
  model = load_model('../trained-model/mnistmodel.h5')
  
  test_labels = test['label']

  test.drop('label', axis = 1, inplace = True)

  test_images = test.values
  test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
  test_images = np.array([i.flatten() for i in test_images])

  #Transforms labels into an array of classes (24 classes) per 8566 rows. The 0 does not belong to the classes, 1 belongs to the class
  label_binrizer = LabelBinarizer()
  test_labels = label_binrizer.fit_transform(test_labels)

  test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

  print('test_images.shape: ', test_images.shape)

  #Resizes the array to 4 dimensions
  imgToPredict = test_images[0].reshape(1,28,28,1)
  print('imgToPredict.shape: ', imgToPredict.shape)

  #Save the array to a file
  #np.save('imgMatriz', lista)

  y_pred = model.predict(imgToPredict)
  print('y_pred: ', y_pred)
  return y_pred
  
if __name__ == "__main__":
  
  from pathlib import Path
  my_model = Path("../trained-model/mnistmodel.h5")
  if not (my_model.exists()):
    trainModel()
  else:
    predictTests()
    
    
    
    
    
    