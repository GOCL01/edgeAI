# -*- coding: utf-8 -*-
"""
########################################################################################################
########################################################################################################
        
    Author       : L.Gonzalez-Carabarin
    Organization : Fontys University of Applied Sciences
    Course       : Mechatronics Lectoraat
    Source Name  : tf_lenet300_q.py
    Description  : Toy example of Post-training quantization of LeNet300

########################################################################################################
########################################################################################################
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from q_layers import *
#tf.compat.v1.disable_eager_execution()

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

# reshape input data (1-D)
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

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


# Parameters
n_epochs = 1


# model creation 
x_ = Input(shape=(x_train.shape[-1]))
x = Dense(300,activation='relu')(x_)
x = Dense(100,activation='relu')(x)
y = Dense(10,activation='softmax')(x)
model = Model(inputs=x_, outputs=y)

#prinitng model summary
model.summary()
  

import callbacks
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# This callback allows to save models based on the best accuracy
callback_model_checkpoint=tf.keras.callbacks.ModelCheckpoint(
  filepath='model_dense.hdf5',
  #filepath='weights.{epoch:02d}.hdf5',
  monitor = 'val_categorical_accuracy',
  verbose = 0,
  save_best_only = True,
  save_weights_only = False,
  save_freq = 'epoch'
)

callbacks = [
             callback_model_checkpoint]

#model compilation 
model.compile(optimizer = optimizer,
             loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

### Model training
### Optimization, use bs=16 instead of 8 leads to faster convergene 
history = model.fit(x=x_train, y=y_train, batch_size=16, epochs=n_epochs, verbose=1,
          callbacks = callbacks,
          validation_data=(x_val,y_val))

### Evaluate last epoch
results = model.evaluate(x_test, y_test, batch_size=16)
print(results)


###############################################################################
###############################################################################
# Loading model
model = tf.keras.models.load_model('model_dense.hdf5')

# Weight quantization
new_weights_l1, scale1 = quantized_w(8, model.layers[1].get_weights())
new_weights_l2, scale2 = quantized_w(8, model.layers[2].get_weights())
new_weights_l3, scale3 = quantized_w(8, model.layers[3].get_weights())

# New model with quantized and dequantized layers
x_ = Input(shape=(x_train.shape[-1]))
x = Dense(300,activation='linear')(x_)
# dequantize
x = dequantized_layer()(x, scale1)
# quantize
x, scale = quantized_layer(8)(x)
x = Activation('relu')(x)
x = Dense(100,activation='linear')(x)
# dequantize
x = dequantized_layer()(x, scale2)
# quantize
x, scale = quantized_layer(8)(x)
x = Activation('relu')(x)
x = Dense(10,activation='linear')(x)
#dequantized
x = dequantized_layer()(x, scale3)
y = Activation('softmax')(x)

# Creating the new model
model = Model(inputs=x_, outputs=y)
model.summary()

model.compile(optimizer = optimizer,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


# Transferring quantized weights
model.layers[1].set_weights(new_weights_l1)
model.layers[5].set_weights(new_weights_l2)
model.layers[9].set_weights(new_weights_l3)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
y_test = to_categorical(y_test)
x_test = x_test/2
### Evaluate last epoch
results = model.evaluate(x_test, y_test, batch_size=16)
print(results)
