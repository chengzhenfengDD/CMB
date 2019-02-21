#Keras Model example
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import plotly.graph_objs as go
from matplotlib.pyplot import cm
from keras.models import Model
import numpy as np
import keras
import h5py

def model_3DCNN(input_size = (20,20,16,1)):

    ## input layer
    input_layer = Input(input_size)

    ## convolutional layers
    conv1 = Conv3D(filters=32, kernel_size=(7, 7, 5), activation='relu', padding = 'valid')(input_layer)
    maxpool1 = MaxPool3D(pool_size=(2, 2, 2),strides =2, padding = 'valid')(conv1)
    conv2 = Conv3D(filters=64, kernel_size=(5, 5, 3), activation='relu', padding = 'valid')(maxpool1)
    conv2 = BatchNormalization()(conv2)
    flatten_layer = Flatten()(conv2)

    ## dense layers : 500 -> 100 -> 2
    ## add dropouts to avoid overfitting / perform regularization
    fc1 = Dense(units=500, activation='relu')(flatten_layer)
    fc1 = Dropout(0.3)(fc1)
    fc2 = Dense(units=100, activation='relu')(fc1)
    fc2 = Dropout(0.3)(fc2)
    output_layer = Dense(units=2, activation='softmax')(fc2)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
