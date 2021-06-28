# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:57:31 2021

@author: PC
"""
#%%

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2



#%%
def CustomCNN(img_height, img_width, img_channel, n_classes, weight_decay):
    input_layer = Input(shape=(img_height, img_width, img_channel))
    x = Conv2D(64, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='elu', name='Conv2D_1')(input_layer)
    x = MaxPooling2D(2)(x)

    x = Conv2D(128, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='elu', name='Conv2D_2')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(256, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='elu', name='Conv2D_3')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(512, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='elu', name='Conv2D_4')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(512, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='elu', name='Conv2D_5')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(512, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='elu', name='Conv2D_6')(x)

    x = Conv2D(512, 3, strides=1, padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
               activation='elu', name='Conv2D_7')(x)

    x = Flatten()(x)

    x = Dense(256, activation='elu')(x)
    x = Dense(256, activation='elu')(x)
    output_layer = Dense(n_classes, activation='softmax')(x)
    model = Model(input_layer, output_layer)
    return model


#%%


