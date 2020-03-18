import cv2 # cv2.__version__ =  4.1
import numpy as np
from utils import params  # file iclude parmater of model
import tensorflow as tf # tf.__version__ == 2.1 
import matplotlib.pyplot as plt 


from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, 
        MaxPooling2D, LeakyReLU,BatchNormalization, Reshape, AveragePooling3D, GlobalAveragePooling3D,
        Input, concatenate, LSTM, TimeDistributed, Bidirectional)

# =========================base model pre-train I3D 
     
params = params.Params()

WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}

def conv3d_bn(x,filters,num_frames,num_row, num_col,padding='same', strides=(1, 1, 1),
        use_bias = False, use_activation_fn = True,use_bn = True,name=None, data_format = 1):

    '''
        Conv3d layer and batchnorm and activation

    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(filters, kernel_size = (num_frames, num_row, num_col),strides=strides,
        padding=padding,use_bias=use_bias,name=conv_name)(x)
    if use_bn:
        if data_format == 0:
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)
    return x

def InceptionV1_3D( input_shape=None,dropout_prob=0.5, mode = 'rgb', data_format = 1):

    '''
        Init model Conv3D base InceptionV1
        Input: 
        + input_shape: input of model include (n_frames, img_width, img_height, n_chanels)
        + dropout
        + mode : rgb | opt use model for input RGB or Optical Flow 
        + data_format: use normalize axis by batchnorm ( for chanel first or chanel last)
    '''

    if data_format == 0:  # chanel first

        channel_axis = 1
    else:

        channel_axis = 4 # chanel last
    if input_shape == None:
        if mode == 'rgb': # if use image have format RGB (with, height, chanel = 3)

            input_shape = (params.n_frames, params.image_shape[0], params.image_shape[1], 3) 
            input_shape = Input(shape = input_shape) # init input with shape 

        else: # if use image have format Optical flow with 2 chanels

            input_shape = (params.n_frames, params.img_size, params.img_size, 2)
            input_shape = Input(shape = input_shape)
    else:
        input_shape = Input(shape = input_shape)

    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(input_shape, 64, 7, 7, 7, strides=(2, 2, 2), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same')

    x = concatenate([branch_0, branch_1, branch_2, branch_3],axis=channel_axis)

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)
   
    model = Model(input_shape, x)           

    # load weights
    if mode == 'rgb':
        weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics'] #rgb_imagenet_and_kinetics flow_imagenet_and_kinetics
        model_name = 'rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
        model.load_weights(downloaded_weights_path)
    else:
        weights_url = WEIGHTS_PATH_NO_TOP['flow_imagenet_and_kinetics'] #rgb_imagenet_and_kinetics flow_imagenet_and_kinetics
        model_name = 'flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
        model.load_weights(downloaded_weights_path)
    
    '''
    Freeze or train all model 
    '''
    # for layer in model.layers:
    #     layer.trainable = False

    # Covert to define shape or classifiaction 
    #  
    x = model.output
    x = AveragePooling3D((1, 7, 7), strides=(1, 1, 1), padding='valid')(x)
    x = Dropout(dropout_prob)(x)
    x = Flatten()(x)
    # x =  Activation('sigmoid')(x)
    x = Dense(params.n_outputI3D)(x) # covert oupt of model I3D have shape (None , 1024)
    x =  Activation('relu')(x)
    model = Model(input_shape, x)
    return model


def get_models_I3D_LSTM():

    i3d_model = Inception_Inflated3d() # build model 
    # i3d_model.summary()

    model = Sequential()
    model.add(TimeDistributed(i3d_model, input_shape = (5, 8, 224, 224, 3)))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    # model.add(Flatten())
    model.add(Dense(101, activation='softmax'))
    model.summary() 
    return model

# get_models()import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, 
        MaxPooling2D, LeakyReLU,BatchNormalization, Reshape, AveragePooling3D, GlobalAveragePooling3D,
        Input, concatenate, LSTM, TimeDistributed, Bidirectional, Conv1D,  MaxPooling1D)
        
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K

from utils import params 
# from utils import params 
# import params 
params = params.Params()
def conv3d_bn(x,filters,num_frames,num_row, num_col,padding='same', strides=(1, 1, 1),
        use_bias = False, use_activation_fn = True,use_bn = True,name=None, data_format = 1):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(filters, kernel_size = (num_frames, num_row, num_col),strides=strides,
        padding=padding,use_bias=use_bias,name=conv_name)(x)
    if use_bn:
        if data_format == 0:
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)
    return x

WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}

def Inception_Inflated3d(input_shape=None,dropout_prob=0.5, mode = 'rgb', data_format = 1):
    if data_format == 0:
        channel_axis = 1
    else:
        channel_axis = 4

    if mode == 'rgb':
        input_shape = (params.n_frames, params.img_size, params.img_size, 3)
        input_shape = Input(shape = input_shape)

    else:
        input_shape = (params.n_frames, params.img_size, params.img_size, 2)
        input_shape = Input(shape = input_shape)
    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(input_shape, 64, 7, 7, 7, strides=(2, 2, 2), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same')

    x = concatenate([branch_0, branch_1, branch_2, branch_3],axis=channel_axis)

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same')

    x = concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis)
    # x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
    # x = Dropout(dropout_prob)(x)

    # x = conv3d_bn(x, args.nclass, 1, 1, 1, padding='same', 
    #             use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    model = Model(input_shape, x)           
    # load weights
    if mode == 'rgb':
        weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics'] #rgb_imagenet_and_kinetics flow_imagenet_and_kinetics
        model_name = 'rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
        model.load_weights(downloaded_weights_path)
    else:
        weights_url = WEIGHTS_PATH_NO_TOP['flow_imagenet_and_kinetics'] #rgb_imagenet_and_kinetics flow_imagenet_and_kinetics
        model_name = 'flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')
        model.load_weights(downloaded_weights_path)
    # for layer in model.layers:
    #     layer.trainable = False
    # add ouput ------------------------------------------------------------------------------------
    x = model.output
    x = AveragePooling3D((1, 7, 7), strides=(1, 1, 1), padding='valid')(x)
    x = Dropout(dropout_prob)(x)
    x = Flatten()(x)
    # x =  Activation('sigmoid')(x)
    x = Dense(1024)(x)
    x =  Activation('relu')(x)
    model = Model(input_shape, x)
    return model


def get_models():

    i3d_model = Inception_Inflated3d()
    # i3d_model.summary()
    model = Sequential()

    # add layer LSTM 
    model.add(TimeDistributed(i3d_model, input_shape = (params.nblocks, params.nframes, params.image_shape[0], params.image_shape[1], 3)))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(256))

    model.add(Dropout(0.2))
    # model.add(Flatten())
    model.add(Dense(params.nclasses, activation='softmax')) 

    model.summary() 
    return model
