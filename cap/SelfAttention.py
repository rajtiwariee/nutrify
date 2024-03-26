# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 08:01:36 2020

@author: Ardhendu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:33:32 2019

@author: Ardhendu
"""
# from keras.layers import Layer
# #from keras import layers
# from keras import backend as K


# import tensorflow as tf
from tensorflow.keras.layers import Layer
# from tensorflow.keras import layers is also a valid import if you need to access other layers
from tensorflow.keras import backend as K

import tensorflow as tf

#from SpectralNormalizationKeras import ConvSN2D

def hw_flatten(x) :
        x_shape = K.shape(x)
        return K.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]
    
class SelfAttention(Layer):
    def __init__(self, filters, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)  # Initialize the parent class first
        self.dim_ordering = K.image_data_format()  # Get the current image data format ('channels_last' or 'channels_first')
        
        # Update the assertion to match the TensorFlow 2.x convention
        assert self.dim_ordering in {'channels_last', 'channels_first'}, "dim_ordering must be in {'channels_last', 'channels_first'}"
        
        self.filters = filters
        
    def build(self, input_shape):
        #self.f = ConvSN2D(self.filters // 8, kernel_size=1, strides=1, padding='same')# [bs, h, w, c']
        #self.g = ConvSN2D(self.filters // 8, kernel_size=1, strides=1, padding='same') # [bs, h, w, c']
        #self.h = ConvSN2D(self.filters, kernel_size=1, strides=1, padding='same') # [bs, h, w, c]
        
        #self.f = layers.Conv2D(self.filters // 8, kernel_size=1, strides=1, padding='same')# [bs, h, w, c']
        #self.g = layers.Conv2D(self.filters // 8, kernel_size=1, strides=1, padding='same') # [bs, h, w, c']
        #self.h = layers.Conv2D(self.filters, kernel_size=1, strides=1, padding='same') # [bs, h, w, c]
        
        
        #self.gamma = tf.get_variable(self.gamma_name, [1], initializer=tf.constant_initializer(0.0))
        self.gamma = self.add_weight(shape=(1,),
                                     name='{}_b'.format(self.name),
                                     initializer='zeros', trainable=True)
        
        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self,x):
        
        assert(len(x) == 4)
        img = x[0]
        f = x[1]
        g = x[2]
        h = x[3]
        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
        
        beta = K.softmax(s)  # attention map
        
        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]        
        o = K.reshape(o, shape=[K.shape(img)[0], K.shape(img)[1], K.shape(img)[2], self.filters])  # [bs, h, w, C]
        #o = K.reshape(o, shape=[K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], self.filters // 2])  # [bs, h, w, C]
        #print(o.shape[0])
        #print(o.shape[1])
        #print(o.shape[2])
        #print(o.shape[3])
        #o = ConvSN2D(self.filters, kernel_size=1, strides=1, padding='same')(o)
        img = self.gamma * o + img
        
        return img
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
