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

class SNPBlockLayer(layers.Layer):
    """
    A layer to calculate LD value based on given LD group files
    Prograssing.
    """
    
    def __init__(self, reference, channels = 8, **kwargs):
        super(SNPBlockLayer, self).__init__(**kwargs)
        self.channels = channels
        self.reference = reference ## An identifing matrix for SNP Blocking (0/1 Matrix)
        
    def build(self, input_shape):
        """
        A weight matrix for SNP weights
        
        """
        self.bweight = self.add_weight(name='Block_weightMatrix', shape=(self.channels,input_shape[1]),
                                  initializer='normal')
        self.built = True

        super(SNPBlockLayer, self).build(input_shape)

    def call(self, x):

        # require constant attention score from attention layer
        # x shape == (batch_size, seq_len)
        # reference shape == (seq_len, LD_len)
        annotation = tf.keras.utils.to_categorical(self.reference)

        #x = tf.squeeze(x)

        extended_X = tf.multiply(x,annotation)
        #tf.einsum("sn,sd->sd",x,annotation) ##got shape == (batch,seq,LD)
        
        extended_LD = tf.multiply(tf.expand_dims(extended_X, axis=2), tf.expand_dims(self.bweight, axis=1))
        #tf.einsum("sl,cs->slc",extended_X,self.bweight) ## (b,s,l) * (channel,s) -> (b,s,l,c)
        
        extended_LD = tf.reduce_sum(extended_LD,axis=1)  #(b,s,l,c) -> (b,sum(s),l,c) -> (b,l,c)
        
        return extended_LD
    
    def get_config(self):
        config = super(SNPBlockLayer, self).get_config()
        config.update({
            'channels':self.channels,
            'reference':self.reference,
            #'built':self.built
            })
            
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
        
    
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

class MultiHead_QKV_BlockAttention(layers.Layer):
    def __init__(self,head_num=1,residual=True, **kwargs):
        super(MultiHead_QKV_BlockAttention, self).__init__(**kwargs)
        self.head_num = head_num
        self.residual = residual
        #self.return_attention = False
        #self.feature_dim = None
        #self.seq_len = None


    def build(self, input_shape):
        #assert len(input_shape[0]) >= 2
        self.return_attention = False
        self.feature_dim = input_shape[0][-1]
        self.seq_len = input_shape[0][1] / self.head_num

        self.wq = self.add_weight(name='query', shape=(self.feature_dim,self.feature_dim),
                                  initializer='normal', trainable=True)
        self.wk = self.add_weight(name='key', shape=(self.feature_dim,self.feature_dim),
                                  initializer='normal', trainable=True)
        self.wv = self.add_weight(name='value', shape=(self.feature_dim,self.feature_dim),
                                  initializer='normal', trainable=True)

        self.built = True
        super(MultiHead_QKV_BlockAttention, self).build(input_shape)

    def call(self, x):
        if type(x) is list and len(x) >= 2:
            X,residual_score = x
        else:
            X = x[0]
            residual_score = 0
        query = tf.tensordot(X,self.wq)
        #tf.einsum('sd,dd->sd',X,self.wq)
        key = tf.tensordot(X,self.wk)
        #tf.einsum('sd,dd->sd',X,self.wk)
        value = tf.tensordot(X,self.wv)
        #tf.einsum('sd,dd->sd',X,self.wv)
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
        if self.residual:
            return tf.add(all_effects,residual_score),tf.add(all_effects,residual_score)

        return all_effects,tf.add(all_effects,residual_score)
    
    def get_config(self):
        config = super(MultiHead_QKV_BlockAttention, self).get_config()
        config.update({
            'head_num':self.head_num,
            'residual':self.residual,
            #'return_attention':self.return_attention,
            #'feature_dim':self.feature_dim,
            #'seq_len':self.seq_len
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        input_shape = input_shape
        if self.residual:
            return [input_shape, input_shape]
        return input_shape

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
        attention_value = tf.einsum('bsnd,nqd->bsqd',exquery,self.wk) #shape = (b,s,q(=s),d)
        attention_score = tf.nn.softmax(attention_value)
        trans_attention_score = tf.transpose(attention_score,perm=[0,1,3,2]) # shape: (b,s,d,q)

        return trans_attention_score,x #shape (batch,seq,embed,seq)

    def compute_output_shape(self, output_shape):
        input_shape = output_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def get_config(self):
        return super(MultiHead_Seq_BlockAttention, self).get_config()    
    
class MultiHead_conv_BlockAttention(layers.Layer):
    def __init__(self,second_embed = None, **kwargs):
        super(MultiHead_conv_BlockAttention, self).__init__(**kwargs)
        #self.head_num = head_num
        self.second_embedding = second_embed

    def build(self, input_shape):

        self.first_embed = input_shape[0][-1]
        self.seq_dim = input_shape[0][1]
        self.value_embed = input_shape[1][-1]
        if self.second_embedding is None:
            self.second_embedding = self.first_embed
        self.wv1 = self.add_weight(name='value', shape=(self.value_embed, self.value_embed),
                                  initializer='normal', trainable=True)
        self.wq = self.add_weight(name='value', shape=(self.seq_dim, self.second_embedding),
                                  initializer='normal', trainable=True)

        self.built = True
        super(MultiHead_conv_BlockAttention, self).build(input_shape)

    def call(self, x):
        attention_score,value = x
        value_v = tf.einsum('bsd,dd->bsd',value,self.wv1)
        value_v = tf.expand_dims(value_v,axis=-1) #bsd1
        #attention_score = tf.transpose(x,perm=[0,3,1,2]) #b,q,d,s -> b,s,q,d
        attention_map = tf.multiply(attention_score, value_v)
        #attention_map = tf.einsum('bsdq,bsdn->bsdq', attention_score, value_v)
        #effect_map = tf.tensordot(attention_map,self.wv2,axes=)
        effect_map = tf.matmul(attention_map,self.wq) #tf.einsum('bsdq,qn->bsdn',attention_map,self.wv2)
        effect_map = tf.transpose(effect_map, perm=[0,2,3,1]) # b,d,n,s

        return effect_map
    
    def compute_output_shape(self, output_shape):
        input_shape = output_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def get_config(self):
        return super(MultiHead_conv_BlockAttention, self).get_config()


class MultiLevel_BlockAttention(layers.Layer):
    """
    LD or multi-level based block attention
    SNP weight = (LDxLD) * (insider LD)
    LD: categorical embedding (dict length, LDs+1, individual SNP grouped as LD 0 (labelled as 1)
    """

    def __init__(self,num_heads=2, **kwargs):
        super(MultiLevel_BlockAttention, self).__init__(**kwargs)
        self.head_num=num_heads

    def build(self, input_shape, annotation=False):
        assert len(input_shape) >= 2
        self.return_attention = False
        self.ld_embedding = input_shape[1][-1]
        self.seq_embedding = input_shape[0][-1]
        self.seq_dim = input_shape[1]
        self.embedding = input_shape[-1]

        self.Wq_ld = self.add_weight(name='Annotation_embedding_q_weights', shape=(self.ld_embedding, self.ld_embedding),
                                  initializer='normal', trainable=True)

        self.Wk_ld = self.add_weight(name='Annotation_embedding_k_weights', shape=(self.ld_embedding, self.ld_embedding),
                                    initializer='normal', trainable=True)

        self.Wv_ld = self.add_weight(name='Annotation_embedding_v_weights', shape=(self.ld_embedding, self.ld_embedding),
                                    initializer='normal', trainable=True)

        self.Wq_seq = self.add_weight(name='Sequence_embedding_q_weights', shape=(self.seq_embedding, self.seq_embedding),
                                   initializer='normal', trainable=True)

        self.Wk_seq = self.add_weight(name='Sequence_embedding_k_weights', shape=(self.seq_embedding, self.seq_embedding),
                                      initializer='normal', trainable=True)

        self.wq = self.add_weight(name='query', shape=(self.embedding, self.embedding),
                                  initializer='normal', trainable=True)
        self.wk = self.add_weight(name='key', shape=(1, self.seq_dim, self.embedding),
                                  initializer='normal', trainable=True)

        self.built = True
        super(MultiLevel_BlockAttention, self).build(input_shape)

    def call(self, x):

        #query = tf.einsum('bsd,dd->bsd', x, self.q_ld)
        #key = tf.einsum('bsd,dd->bsd', x, self.k_ld)
        #value = tf.einsum('bsd,dd->bsd', x, self.v_ld)

        query = tf.tensordot(x, self.q_ld, axes=(-1, 0))
        key = tf.tensordot(x, self.k_ld, axes=(-1, 0))
        value = tf.tensordot(x, self.v_ld, axes=(-1, 0))

        # q,k,v shape == (batch_size, seq_len, d_model)

        query = tf.stack(tf.split(query, self.head_num, axis=2))
        key = tf.stack(tf.split(key, self.head_num, axis=2))
        value = tf.stack(tf.split(value, self.head_num, axis=2))

        inner_product = tf.matmul(query, key, transpose_b=True)
        attention_score = tf.nn.softmax(inner_product)

        effect = tf.matmul(attention_score, value)

        all_effects = tf.concat(tf.split(effect, self.head_num, ), axis=-1)
        all_effects = tf.squeeze(all_effects, axis=0)  # (batch_size, seq_len, d_model)\

        return all_effects, x  # shape (batch,seq,embed,seq)

    def compute_output_shape(self, output_shape):
        input_shape = output_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def get_config(self):
        return super(MultiLevel_BlockAttention, self).get_config()



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
        input_x = input
        if input.shape[-1] != X.shape[-1]:
            filter_n = X.shape[-1]
            input_x = layers.Dense(filter_n,activation='relu')(X)
        out = layers.Add()([X, input_x])
        #out = activation(out)
        return activation(out)
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