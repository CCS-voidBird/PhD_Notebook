#import dask.array
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
"""
This file contains custom layers for ML_composer
"""

"""
Positional encoding layer
"""
def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
       #x = K.squeeze(x, axis=-1)
        return K.squeeze(K.dot(x, K.expand_dims(kernel)),axis=-1)
    else:
        return K.dot(x, kernel)

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
        #return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        return self.pos_encoding[:, :tf.shape(inputs)[1], :]

class BlockAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(BlockAttention, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.return_attention = False
        #attention_dim = [input_shape[-1]]
        #amount_size = input_shape[1:-1]
        #input_shape = tf.TensorShape(input_shape)
        #input_channel = self._get_input_channel(input_shape)
        self.filters = input_shape[-1]
        self.u = self.add_weight(name='Block_extension', shape=(self.filters,input_shape[-2]),
                                  initializer='ones', trainable=False)
        self.Wa = self.add_weight(name='Attention_context_vector', shape=(self.filters,input_shape[-2], input_shape[-2]),
                                 initializer='normal', trainable=True)
        self.We = self.add_weight(name='effect_context_vector', shape=(self.filters,input_shape[-2], input_shape[-2]),
                                  initializer='normal', trainable=True)
        #self.u = self.add_weight(shape=(input_shape[-2],),initializer='normal',name='Attention_u')
        #self.W2 = self.add_weight(name='Attention_weight', shape=(amount_size,attention_dim), initializer='normal', trainable=True)
        self.built = True
        super(BlockAttention, self).build(input_shape)

    def call(self, x):
        # require constant attention score from attention layer
        # x shape == (batch_size, seq_len,seq_len, d_model
        e = K.dot(x, self.u)
        att = e * self.Wa
        #sum by features
        #eff = K.sum(e, axis=1)
        att = K.softmax(att)
        eff = att * e * self.We
        #sum by time
        eff = K.sum(eff, axis=1, keepdims=True)
        #e = K.batch_dot(K.dot(x, self.Wa), K.permute_dimensions(x, (0, 2, 1)))
        #uit = K.expand_dims(uit, axis=-1)
        #ait = dot_product(uit, self.u)

        #a = K.exp(ait)

        #a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        #e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        #a = e / K.sum(e, axis=-1, keepdims=True)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        #v = K.batch_dot(a, x)

        if self.return_attention:
            return [eff, att]
        return eff

class MultiHead_BlockAttention(layers.Layer):
    def __init__(self,head_num, **kwargs):
        super(MultiHead_BlockAttention, self).__init__(**kwargs)
        self.head_num = head_num


    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.return_attention = False
        self.filters = input_shape[-1]
        self.seq_len = input_shape[1]

        self.wq = self.add_weight(name='query', shape=(self.filters,self.seq_len*self.head_num),
                                  initializer='normal', trainable=True)
        self.wk = self.add_weight(name='key', shape=(self.filters,self.seq_len*self.head_num),
                                  initializer='normal', trainable=True)
        self.wv = self.add_weight(name='value', shape=(self.filters,self.seq_len*self.head_num),
                                  initializer='normal', trainable=True)

        self.built = True
        super(MultiHead_BlockAttention, self).build(input_shape)

    def call(self, x):

        query = tf.tensordot(x, self.wq, axes=(-1,0))
        key = tf.tensordot(x, self.wk, axes=(-1,0))
        value = tf.tensordot(x, self.wv, axes=(-1,0))

        # q,k,v shape == (batch_size, seq_len, d_model)

        query = tf.stack(tf.split(query, self.head_num, axis=2))
        key = tf.stack(tf.split(key, self.head_num, axis=2))
        value = tf.stack(tf.split(value, self.head_num, axis=2))

        inner_product = tf.matmul(query, key, transpose_b=True)
        attention_score = tf.nn.softmax(inner_product)

        effect = tf.matmul(attention_score, value)

        all_effects = tf.concat(tf.split(effect, self.head_num,), axis=-1)
        all_effects = tf.squeeze(all_effects, axis=0) # (batch_size, seq_len, d_model)\

        return all_effects

class MultiHead_QK_BlockAttention(layers.Layer):
    def __init__(self,head_num=1, **kwargs):
        super(MultiHead_QK_BlockAttention, self).__init__(**kwargs)
        self.head_num = head_num


    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.return_attention = False
        self.feature_dim = input_shape[-1]
        self.seq_len = input_shape[1] / self.head_num

        self.wq = self.add_weight(name='query', shape=(self.feature_dim,self.feature_dim),
                                  initializer='normal', trainable=True)
        self.wk = self.add_weight(name='key', shape=(self.feature_dim,self.feature_dim),
                                  initializer='normal', trainable=True)
        self.wv = self.add_weight(name='value', shape=(self.feature_dim,self.feature_dim),
                                  initializer='normal', trainable=True)

        self.built = True
        super(MultiHead_QK_BlockAttention, self).build(input_shape)

    def call(self, x):

        query = tf.einsum('bsd,dd->bsd',x,self.wq)
        key = tf.einsum('bsd,dd->bsd',x,self.wk)
        value = tf.einsum('bsd,dd->bsd',x,self.wv)
        #value = tf.tensordot(x, self.wv, axes=(-1,0))

        # q,k,v shape == (batch_size, seq_len, d_model)

        query = tf.stack(tf.split(query, self.head_num, axis=2),axis=1)
        key = tf.stack(tf.split(key, self.head_num, axis=2),axis=1)
        value = tf.stack(tf.split(value, self.head_num, axis=2),axis=1)

        inner_product = tf.matmul(query, key, transpose_b=True)/tf.math.sqrt(self.seq_len)
        attention_score = tf.nn.softmax(inner_product)

        effect = tf.matmul(attention_score, value)

        all_effects = tf.concat(tf.split(effect, self.head_num,axis=1), axis=-1)
        all_effects = tf.squeeze(all_effects, axis=1) # (batch_size, seq_len, d_model)

        return all_effects

class MultiHead_Seq_BlockAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(MultiHead_Seq_BlockAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.return_attention = False
        self.seq_dim  = input_shape[1]
        self.embedding = input_shape[-1]

        self.wq = self.add_weight(name='query', shape=(self.embedding,self.embedding),
                                  initializer='normal', trainable=True)
        self.wk = self.add_weight(name='key', shape=(1,self.seq_dim,self.embedding),
                                  initializer='normal', trainable=True)
        

        self.built = True
        super(MultiHead_Seq_BlockAttention, self).build(input_shape)

    def call(self, x):
        
        query = tf.einsum('bsd,dd->bsd',x,self.wq)
        exquery = tf.expand_dims(query,axis=2) #shape = (b,s,1,d)
        attention_value = tf.einsum('bsqd,qsd->bssd',exquery,self.wk) #shape = (b,s,s,d)
        #value = tf.einsum('bsd,dd->bsd',x,self.wv)
        #value = tf.tensordot(x, self.wv, axes=(-1,0))
        #key_trans = tf.transpose(key,perm=[0,2,1])
        attention_score = tf.nn.softmax(attention_value)
        trans_attention_score = tf.transpose(attention_score,perm=[0,2,3,1])

        return attention_score,self.embedding #shape (batch,seq,embed,seq)

    def compute_output_shape(self, output_shape):
        input_shape = output_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def get_config(self):
        return super(MultiHead_Seq_BlockAttention, self).get_config()    
    
class MultiHead_conv_BlockAttention(layers.Layer):
    def __init__(self,head_num=1,second_embed = None, **kwargs):
        super(MultiHead_conv_BlockAttention, self).__init__(**kwargs)
        self.head_num = head_num
        self.second_embedding = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.return_attention = False
        self.seq_dim  = input_shape[1]
        self.embedding = input_shape[-1]
        self.kernel_size = input_shape[-1]
        if self.second_embedding is None:
            self.second_embedding = self.embedding

        self.wv = self.add_weight(name='value', shape=(self.seq_dim,self.embedding),
                                  initializer='normal', trainable=True)

        self.built = True
        super(MultiHead_conv_BlockAttention, self).build(input_shape)

    def call(self, x):
        
        #attention_score = tf.transpose(x,perm=[0,3,1,2]) #b,q,d,s -> b,s,q,d
        effect_map = tf.einsum('bqds,qd->bqds',x,self.wv)
        


        return effect_map
    
    def compute_output_shape(self, output_shape):
        input_shape = output_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def get_config(self):
        return super(MultiHead_conv_BlockAttention, self).get_config()

##Create a self attention layer with weights
class PosAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(PosAttention, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) >= 2
        attention_dim = input_shape[-1]
        amount_size = input_shape[-1]
        #input_shape = tf.TensorShape(input_shape)
        #input_channel = self._get_input_channel(input_shape)
        self.W1 = self.add_weight(name='Attention_weight', shape=(amount_size,attention_dim),
                                  initializer='normal', trainable=True)

        self.W2 = self.add_weight(name='Attention_weight', shape=(amount_size,attention_dim),
                                  initializer='normal', trainable=True)
        self.V = layers.Dense(1, use_bias=False)
        self.built = True
        super(PosAttention, self).build(input_shape)

    def call(self, x, pos):
        # require constant attention score from attention layer
        # x shape == (batch_size, seq_len,seq_len, d_model)
        pos_attention = dot_product(pos, self.W1)

        tanh_attention = tf.nn.tanh(pos_attention)
        alpha = tf.nn.softmax(dot_product(tanh_attention, self.W2), axis=1)

        epi = dot_product(tf.nn.softmax(tanh_attention, axis=1), x)

        # score shape == (batch_size, seq_len, seq_len, 1)


        return K.sum(epi, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        return super(PosAttention, self).get_config()


def residual_fl_block(input, width, activation=layers.ReLU(),downsample=False):
    #residual fully connected layers block
    X = layers.Dense(width)(input)
    if activation == layers.ReLU():
        X = layers.BatchNormalization()(X)

    if downsample:
        if input.shape[-1] != X.shape[-1]:
            filter_n = X.shape[-1]
            input_x = layers.Conv1D(filter_n,1,1,padding='same')(X)
        out = layers.Add()([X, input_x])
        #out = activation(out)
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