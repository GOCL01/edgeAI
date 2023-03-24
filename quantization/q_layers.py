# -*- coding: utf-8 -*-
"""
########################################################################################################
########################################################################################################
        
    Author       : L.Gonzalez-Carabarin
    Organization : Fontys University of Applied Sciences
    Course       : Mechatronics Lectoraat
    Source Name  : q_layers.py
    Description  : quantized layers

########################################################################################################
########################################################################################################
"""


import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


"""functions"""
def quantized_w(num_bits, x):
    """
    This functions implements quantization of regular multidimensional numpy arrays, 
    targeting model weighst and bias.
    num_bits : bit representation after quantization 
    x        : input (2-dimensional list containing weights and bias)
    """
    
    qmin = 0.
    qmax = 2.**num_bits - 1.
    
    max_val_1 = np.max(np.abs(x[0]))
    max_val_2 = np.max(np.abs(x[1]))
    max_val = np.max([max_val_1, max_val_2]) #max value of x
    
    scale = 2*max_val / (qmax - qmin) # calculating scale
    q_x_w =  x[0] / scale # scaling weights
    q_x_b =  x[1] / scale # scaling bias

    q_x_w = q_x_w.clip(-127, 127).round() #rounding to the nearest integer (w)
    q_x_b = q_x_b.clip(-128, 127).round() #rounding to the nearest integer (b)
    q_x = [q_x_w , q_x_b]

    return q_x, scale


"""classes"""
class quantized_layer(Layer): 
    
    """
    This class is an extension of tensorflow.keras.layers.Layer and implements
    a quantization function.
    arguments
    num_bits : bit representation after quantization 
    input 
    x : tensor input 
    """

    def __init__(self,
                 num_bits,
                 name = None,
                 **kwargs):
        self.num_bits = num_bits
        super(quantized_layer, self).__init__(name=name, **kwargs) 
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_bits':self.num_bits
        })
        return config

    def build(self, input_shape): 
        super(quantized_layer, self).build(input_shape) 
 
    def call(self, x):

        qmin = 0.
        qmax = 2.**self.num_bits - 1.

        t = tf.math.maximum(tf.math.reduce_min(x), tf.math.reduce_max(x))
        max_val = tf.math.abs(t) #max value of x

        scale = 2*max_val / (qmax - qmin)  # calculating scale

        q_x = x/scale # scaling inputs

        q_x = tf.round(tf.clip_by_value(q_x, -127, 127))  #rounding to the nearest integer

        # Dummy tensor to avoid function discontinuting 
        tensor_2 = 2*tf.keras.backend.hard_sigmoid(x)-1.0
        tensor_2 = scale*tensor_2
        # Stopping gradient to avoid function discontinuting 
        q_x_ = tf.stop_gradient(q_x-tensor_2)+tensor_2

        
        return q_x_, scale

class dequantized_layer(Layer):
    
    """
    This class is an extension of tensorflow.keras.layers.Layer and implements
    a dequantization function.
    input 
    x : tensor input 
    scale : scale used in previous layer
    """

    def __init__(self,
                 name = None,
                 **kwargs):
        super(dequantized_layer, self).__init__(name=name, **kwargs) 
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
        })
        return config

    def build(self, input_shape): 
        super(dequantized_layer, self).build(input_shape) 
 
    def call(self, x, scale):

        q_x = tf.math.multiply(x,scale)

        return q_x
    
