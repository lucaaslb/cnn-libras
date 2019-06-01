from keras.utils import to_categorical, plot_model 
from keras.models import load_model


model_name = ''
model = load_model('models/'+ model_name)

plot_model(model, to_file='models/image/'+model_name'.png', show_shapes = True)
