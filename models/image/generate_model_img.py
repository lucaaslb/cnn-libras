from keras.utils import to_categorical, plot_model 
from keras.models import load_model

model = load_model('models/cnn_model_LIBRAS_20190528_001136.h5')

plot_model(model, to_file='models/image/1.png', show_shapes = True)