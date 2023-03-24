# -*- coding: utf-8 -*-
"""
########################################################################################################
########################################################################################################
        
    Author       : L.Gonzalez-Carabarin
    Organization : Fontys University of Applied Sciences
    Course       : MINOR Adaptive Robotics
    Source Name  : tflite_lenet5_quant.py
    Description  : Main file for MNIST using Lenet5 (lite model)

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

x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
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


###############################################################################
## converting model to tflite (quantized)

model = tf.keras.models.load_model('model_dense_lenet5.hdf5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)


# convert to tflite - model before quantization
tflite_model = converter.convert()
# Save the model
open ("model.tflite" , "wb") .write(tflite_model)

# use quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Save the model
open ("model_lenet5_q.tflite" , "wb") .write(tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path="model_lenet5_q.tflite")
interpreter.allocate_tensors()

# Get input and output tensors (optional)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
x_test_ = x_test/255
input_shape = input_details[0]['shape']
#input_data = np.random.random_sample(input_shape), dtype=np.float32)
input_data = x_test[0].reshape(1,28,28).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
 # Run inference.
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
    
counter = 0
y_pred = np.zeros((10000,10))
for i in range(0, 9999):
    input_data = x_test[i].reshape(1,28,28).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_pred[i,:] = output_data
    if np.sum(np.round(output_data)-y_test[i])==0:
        counter = counter+1 # TP
    #print(output_data)

print('accuracy : ', counter/10000)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))