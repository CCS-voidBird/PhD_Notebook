from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd

"""
This file contains custom layers for ML_composer
"""

"""
Positional encoding layer
"""
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=np.arange(position)[:, np.newaxis],
            i=np.arange(d_model)[np.newaxis, :],
            d_model=d_model)
        # apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

##Create a self attention layer with weights
class SelfAttention(layers.Layer):
    def __init__(self, units,dropout=0, **kwargs):
        super(SelfAttention, self).__init__()
        self.Wq = layers.Dense(units, use_bias=False)
        self.Wk = layers.Dense(units, use_bias=False)
        self.Wv = layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, queries, keys, values, **kwargs):
        q = self.Wq(queries)
        k = self.Wk(keys)
        v = self.Wv(values)
        # calculate attention using softmax
        score = tf.matmul(q, k, transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(self.dropout(attention_weights), v)
        return output

def residual_fl_block(input, width, activation=layers.ReLU(),downsample=False):
    #residual fully connected layers block
    X = layers.Dense(width)(input)
    if activation == layers.ReLU():
        X = layers.BatchNormalization()(X)

    if downsample:
        out = layers.Add()([X, input])
        out = activation(out)
        return out
    else:
        X = activation(X)
        return X



def residual_conv1D_block(input,filters,kernel_size,activation=layers.ReLU(),downsample=False):
    X = layers.Conv1D(filters, kernel_size, padding="same")(input)
    X = layers.BatchNormalization()(X)
    X = activation(X)

    X1 = layers.Conv1D(filters, kernel_size, padding="same")(X)
    X1 = layers.BatchNormalization()(X1)

    if downsample:
        input = layers.Conv1D(filters, 1, padding="same")(input)
        input = layers.BatchNormalization()(input)
        out = layers.Add()([X1, input])
    else:
        out = layers.Add()([X1])

    out = activation(out)
    return out