# -*- coding: utf-8 -*-
"""
########################################################################################################
########################################################################################################
        
    Author       : L.Gonzalez-Carabarin
    Organization : Fontys University of Applied Sciences
    Course       : Mechatronics Lectoraat
    Source Name  : dummy_dense_lenet300.py
    Description  : Toy model to generate a h5 file in order to be further compressed.

########################################################################################################
########################################################################################################
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

seed(1)

########################################################################################################
########################################################################################################
# Activate the following lines for GPU's usage 

if False:
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Currently, memory growth needs to be the same across GPUs
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    tf.config.experimental.set_visible_devices(gpus[0],'GPU')


########################################################################################################
########################################################################################################
#%% Load data  


# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape input data

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)


#x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

x_val = x_train[-5000:,:]
x_train = x_train[:-5000,:]

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

y_val = y_train[-5000:,:]
y_train = y_train[:-5000,:]




########################################################################################################
########################################################################################################
#%% Define sparseConnect Model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Activation


# Parameters
n_epochs = 1

#model creation
x_ = Input(shape=(x_train.shape[-1]))
x = Dense(300,activation='relu')(x_)
x = Dense(100,activation='relu')(x)
y = Dense(10,activation='softmax')(x)
model = Model(inputs=x_, outputs=y)

model.summary()
  
########################################################################################################
########################################################################################################
#%% Start training
import callbacks
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# Saving model considering best accuracy
callback_model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
  filepath='model_dense_lenet300.hdf5',
  #filepath='weights.{epoch:02d}.hdf5',
  monitor = 'val_categorical_accuracy',
  verbose = 0,
  save_best_only = True,
  save_weights_only = False,
  save_freq = 'epoch'
)

#Creating list of callbacks
callbacks = [
             callback_model_checkpoint]

# model compilation 
model.compile(optimizer = optimizer,
             loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

### Optimization, use bs=16 instead of 8 leads to faster convergene
history = model.fit(x=x_train, y=y_train, batch_size=16, epochs=n_epochs, verbose=1,
          callbacks = callbacks,
          validation_data=(x_val,y_val))

### Evaluate last epoch
results = model.evaluate(x_test, y_test, batch_size=1)
print(results)

#metrics
y_pred = model.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
