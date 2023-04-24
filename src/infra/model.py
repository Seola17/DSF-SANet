import math

import keras
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K
from keras import layers
from keras.layers import Dense
from keras.models import Model
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import *


def inorm_cnn_module(extended_segment_RRI_RPA):
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=11, strides=1, padding='same')(extended_segment_RRI_RPA)
    
    # conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = keras.layers.BatchNormalization(axis=1)(conv1)

    conv1 = keras.layers.Activation('relu')(conv1)
    conv1 = keras.layers.Dropout(rate=0.2)(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=5)(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=11, strides=1, padding='same')(conv1)
    
    #conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = keras.layers.BatchNormalization(axis=1)(conv2)

    conv2 = keras.layers.Activation('relu')(conv2)
    conv2 = keras.layers.Dropout(rate=0.2)(conv2)
    return conv2


def residual_block(input_layer):
    layer = keras.layers.Conv1D(filters=32, kernel_size=11, strides=1, padding='same', activation=None)(
        input_layer)
    layer = keras.layers.Activation("relu")(layer)
    layer = keras.layers.Conv1D(filters=64, kernel_size=11, strides=1, padding='same', activation=None)(layer)
    layer = keras.layers.Add()([layer, input_layer])
    layer = keras.layers.Activation("relu")(layer)
    return layer


def resnet_module(original_segment_RRI_RPA):
    current_layer = keras.layers.Conv1D(filters=64, kernel_size=11, strides=1, padding='same')(original_segment_RRI_RPA)
    current_layer = keras.layers.Activation("relu")(current_layer)
    for i in range(16):
        current_layer = residual_block(current_layer)
    return current_layer


def eca_block(feature, feature_with_adjacent_segment, b=1, gama=2):
    in_channel = feature.shape[-1]
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))

    if kernel_size % 2:
        kernel_size = kernel_size
    else:
        kernel_size = kernel_size + 1

    x = layers.GlobalAveragePooling1D()(feature)

    x = layers.Reshape(target_shape=(in_channel, 1))(x)

    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)

    x = tf.nn.sigmoid(x)

    x = layers.Reshape((1, in_channel))(x)

    outputs = layers.multiply([feature_with_adjacent_segment, x])

    return outputs


def create_classifier():
    original_segment_RRI_RPA = keras.Input(shape=(180, 2))
    extended_segment_RRI_RPA = keras.Input(shape=(900, 2))

    feature_original_segment = resnet_module(original_segment_RRI_RPA)
    feature_extended_segment = inorm_cnn_module(extended_segment_RRI_RPA)

    feature_concatenated = keras.layers.concatenate([feature_original_segment, feature_extended_segment], axis=-1)

    feature_fusion = eca_block(feature_concatenated, feature_concatenated)

    feature_flatten = layers.Flatten()(feature_fusion)

    fc = Dense(1, activation=None, name="fc_Layer")(feature_flatten)

    output = keras.layers.Activation('sigmoid')(fc)

    model = Model(inputs=[original_segment_RRI_RPA, extended_segment_RRI_RPA], outputs=output)
    return model
