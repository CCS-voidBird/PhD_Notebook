#import dask.array\
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_core as keras
from keras import layers, activations
#import tensorflow_probability as tfp
from keras import backend as K
from keras import layers,utils
import tensorflow as tf
import numpy as np
import pandas as pd
#import tensorflow.keras.backend as K
tf.config.experimental_run_functions_eagerly(True)

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

def is_label_valid(labels):
    """Returns a boolean `Tensor` for label validity."""
    #labels = tf.convert_to_tensor(value=labels)
    #labels = tf.cast(labels,dtype=tf.float32)
    #print(labels.dtype)
    return tf.greater_equal(labels, tf.constant(0.0))

def calculate_ordinal_loss(y_true, y_pred):
    pass

def add_normalization(x=None,x1=None,norm_switch=False,activation='relu'):
    x = layers.Dropout(0.2)(x)
    if norm_switch is True:
        x = layers.Add()([x,x1])
        x = layers.BatchNormalization()(x)
    #x = layers.Activation(activation)(x)
    return x

class BinaryConversionLayer(layers.Layer):
    def __init__(self, condition, **kwargs):
        super(BinaryConversionLayer, self).__init__(**kwargs)
        self.condition = condition

    def call(self, inputs):
        binary_array = tf.where(self.condition(inputs), tf.ones_like(inputs), tf.zeros_like(inputs))
        heter_array = tf.stack([inputs,binary_array],axis=2)
        #heter_array = tf.expand_dims(heter_array,axis=-1)
        return heter_array

    def get_config(self):
        config = super(BinaryConversionLayer, self).get_config()
        config.update({'condition': self.condition})
        return config

class Ordinal_loss:
    def __init__(self,num_classes):
        self.num_classes = num_classes
        self.loss = self._calculate_loss

    def _to_classes(self,labels,mask):
        one_to_n = tf.range(1, self.num_classes + 1, dtype=tf.float32)
        unsqueezed = tf.repeat(
            tf.expand_dims(labels, axis=2), self.num_classes, axis=-1)
        ordinals = tf.where(unsqueezed >= one_to_n, tf.ones_like(unsqueezed), 0.0)
        return tf.where(tf.expand_dims(mask, axis=-1), ordinals, 0.0)

    def is_label_valid(self,labels):
        """Returns a boolean `Tensor` for label validity."""
        #labels = tf.convert_to_tensor(value=labels)
        return tf.greater_equal(labels, 0.)

    def _calculate_loss(self,labels, logits):

        if logits.shape.rank != 3:
            raise ValueError('Predictions for ordinal loss must have rank 3.')

        mask = is_label_valid(labels)
        labels = tf.where(mask, labels, 0.0)
        logits = tf.where(tf.expand_dims(mask, -1), logits, 0.0)###
        ordinals = self._to_classes(labels, mask)
        losses = tf.where(
            tf.expand_dims(mask, -1),
            tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
                labels=ordinals,
                logits=logits),
            0.0)
        return tf.reduce_sum(losses, axis=-1), tf.cast(mask, dtype=tf.float32)

class Cor_mse_loss:

    def __init__(self):
        self.loss = self._calculate_loss

    def _calculate_loss(self,predictions, observations):
        # Calculate the pearson's correlation
        #correlation = tfp.stats.correlation(predictions, observations, sample_axis=0, event_axis=None)
        correlation = keras.ops.correlate(predictions, observations,mode='valid')
        # Got diag cor for correlation matrix
        #correlation = tf.linalg.diag_part(correlation)
        #print(correlation)
        # Calculate the MSE
        mse = tf.reduce_mean(tf.square(predictions - observations), axis=1)
        # Calculate the loss
        loss = 1.0 if tf.math.is_nan(correlation) else tf.subtract(1.0, correlation)
        loss = tf.add(loss, mse)
        return loss

class Var_mse_loss:
    def __init__(self):
        self.loss = self._calculate_loss

    def _calculate_loss(self,predictions, observations):
        # Calculate the variance of the predictions
        variance = tf.math.reduce_variance(predictions, axis=1)
        # Calculate the MSE
        mse = tf.reduce_mean(tf.square(predictions - observations), axis=1)
        # calculate the obv variance
        obs_variance = tf.math.reduce_variance(observations, axis=1)
        # Calculate the abs var part loss
        var_loss = 0.1 * tf.abs(tf.subtract(variance, obs_variance))
        # return adjust loss of mse and varloss
        return tf.add(mse, var_loss)

class R2_score_loss:
    def __init__(self):
        self.loss = self._calculate_loss

    def _calculate_loss(self,predictions, observations):
        # Calculate R2 scores as loss if minus then use MSE
        total_error = tf.reduce_sum(tf.square(tf.subtract(observations, tf.reduce_mean(observations))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(observations, predictions)))
        R_squared = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
        mse_loss = tf.reduce_mean(tf.square(predictions - observations), axis=1)
        loss = R_squared if R_squared > 0.0 else mse_loss
        return loss

class bounded_mse_loss:
    
    def __init__(self,max_correlation=0.6):
        self.loss = self._calculate_loss
        self.max_correlation = max_correlation

    def _calculate_loss(self, y_true, y_pred):
        mse = keras.losses.mean_squared_error(y_true, y_pred)
        correlation = keras.ops.abs(keras.ops.correlate(y_true, y_pred,mode='valid'))
        penalty = tf.maximum(0.0, correlation - self.max_correlation)
        return mse + 10.0 * penalty  # Adjust penalty weight as needed

class HarmonicLayer(layers.Layer):
    def __init__(self,numTrait=1, use_bias=True, **kwargs):
        super(HarmonicLayer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.numTrait = numTrait

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.numTrait,input_shape[-1]),
            initializer='glorot_uniform',
            name='Harmonic kernel',
            trainable=True
        )
        #self.bias = self.add_weight(
        #    shape=(self.numTrait,),
        #    initializer='normal',
        #    name='Harmonic bias', 
        #    trainable=True
        #) if self.use_bias else None

    def call(self, inputs):
        #calculate the Euclidean distance between w and x
        harmonic = tf.math.square(tf.subtract(inputs, self.kernel))
        sum_harmonic = tf.reduce_sum(harmonic, axis=-1)
        #if self.use_bias:
        #    logits = tf.math.reciprocal(tf.add(sum_harmonic,self.bias))
        #get reciprocal of the sum
        logits = tf.math.reciprocal(sum_harmonic)
        return logits


    def get_config(self):
        config = super(HarmonicLayer, self).get_config()
        config = {'numTrait': self.numTrait,
                  'use_bias': self.use_bias}
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class FeatureSelectionLayer(layers.Layer):
    def __init__(self, l1_lambda=0.01, **kwargs):
        super().__init__(**kwargs)
        self.l1_lambda = l1_lambda
        
    def build(self, input_shape):
        self.feature_selector = layers.Dense(input_shape[-1])
        self.sparse_gate = layers.Activation('sigmoid')
        
    def call(self, inputs):
        feature_weights = self.sparse_gate(self.feature_selector(inputs))
        weighted_features = inputs * feature_weights
        # Add L1 regularization
        self.add_loss(self.l1_lambda * tf.norm(feature_weights, ord=1))
        return weighted_features

class OrdinalOutputLayer(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes


    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.num_classes),
            initializer='glorot_uniform',
            name='Ordinal kernel'
        )
        self.bias = self.add_weight(
            shape=(self.num_classes,),
            initializer='zeros',
            name='Ordinal bias'
        )

    def call(self, inputs):
        logits = tf.matmul(inputs, self.kernel) + self.bias
        logits = tf.reshape(logits,(-1,1,self.num_classes))
        probabilities = tf.nn.sigmoid(logits)
        #scores = tf.argmax(probabilities,axis=1,output_type=tf.float32)
        #scores = tf.cast(scores,dtype=tf.float32)
        #scores = tf.expand_dims(probabilities,axis=1)
        scores = probabilities
        #probabilities = tf.argmax(probabilities,axis=-1)
        #scores = tf.reduce_sum(probabilities,axis=1)
        #scores = tf.expand_dims(scores,axis=-1)
        return scores

class Expression_level_encoding(layers.Layer):

    def __init__(self,args, **kwargs):
        super(Expression_level_encoding, self).__init__(**kwargs)
        self.initializer = keras.initializers.RandomNormal(mean=1., stddev=1)
        self.initializerBias = keras.initializers.RandomNormal(mean=0., stddev=1)
        self.encoder_width = args.embedding
        self.condition = lambda x: x == 1.0

    def build(self, input_shape):
        
        self.expression_level = self.add_weight(
            shape=(input_shape[1],self.encoder_width),initializer=self.initializer,name='expression_level'
        )

        self.decodingWeight1 = self.add_weight(
            shape=(self.encoder_width,self.encoder_width),initializer=self.initializer,name='decodingWeight'
        )



    def call(self, inputs):
        ## create one like array for inputs
        expression = tf.ones(shape=(inputs.shape[-2],self.encoder_width),dtype=tf.float32)
        dominant = tf.where(self.condition(inputs), tf.ones_like(inputs), tf.zeros_like(inputs))

        #X = layers.Conv1D(1, kernel_size=10, strides=1, padding='same', activation='relu',use_bias=True)(input1)
        #X = layers.Add()([X, input1])
      
        expression_encoding = tf.multiply(expression,self.expression_level)*tf.constant(np.pi)*tf.constant(0.5)
        #dominant_expression_encoding = tf.multiply(dominant,self.expression_level)*tf.constant(np.pi)*tf.constant(0.5)
        #expression_encoding = layers.Dense(16,activation='linear',use_bias=False)(expression_encoding)

        X = tf.matmul(expression_encoding,self.decodingWeight1)

        
        #X = tf.reduce_mean(X,axis=-1)
        
        X1 = tf.math.sin(X)
        X1 = tf.multiply(inputs,X1)

        X2 = tf.math.sin(X)
        X2 = tf.multiply(dominant,X2)

        return tf.add(X1,X2)

class AddingLayer_with_bias(layers.Layer):

    def __init__(self,**kwargs):
        super(AddingLayer_with_bias, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.Mbias = self.add_weight(
            shape=(1,),name='pesudo_mean_bias'
        )

    def call(self, inputs):
        ## sum all the inputs and bias

        X = tf.add(inputs,self.Mbias)
        return X


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
    
    def __init__(self, channels = 8, **kwargs):
        super(SNPBlockLayer, self).__init__(**kwargs)
        self.channels = channels
        #self.reference = reference ## An identifing matrix for SNP Blocking (0/1 Matrix)
        
    def build(self, input_shape):
        """
        A weight matrix for SNP weights
        
        """
        self.bweight = self.add_weight(name='Block_weightMatrix', shape=(input_shape[1],self.channels),
                                  initializer='normal')
        self.built = True

        super(SNPBlockLayer, self).build(input_shape)

    def call(self, x,annotation):

        # require constant attention score from attention layer
        # x shape == (batch_size, seq_len)
        # reference shape == (seq_len, LD_len)

        #x = tf.squeeze(x)

        LD_mask = is_label_valid(annotation)
        annotation = tf.where(LD_mask, annotation, 0.0) #See if it works
        extended_X = tf.multiply(x,annotation)
        #tf.einsum("sn,sd->sd",x,annotation) ##got shape == (batch,seq,LD)
        
        extended_LD = tf.multiply(tf.expand_dims(extended_X, axis=-1), tf.expand_dims(self.bweight, axis=1))
        #tf.einsum("sl,cs->slc",extended_X,self.bweight) ## (b,s,l,1) * (s,1,channel) -> (b,s,l,c)
        
        extended_LD = tf.reduce_sum(extended_LD,axis=1)  #(b,s,l,c) -> (b,sum(s),l,c) -> (b,l,c)
        
        return extended_LD
    
    def get_config(self):
        config = super(SNPBlockLayer, self).get_config()
        config.update({
            'channels':self.channels,
            #'reference':self.reference,
            #'built':self.built
            })
            
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = BlockAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    

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

    def __init__(self,num_heads=1,return_attention=True,annotation=False,epi_genomic=False,use_bias=True,**kwargs):
        super(MultiLevel_BlockAttention, self).__init__(**kwargs)
        self.head_num=num_heads
        self.return_attention = return_attention
        self.annotation = annotation
        self.use_bias=use_bias
        self.epi_genomic = epi_genomic


    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = tf.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = tf.reshape(x, (batch_size, seq_len, head_num, head_dim))
        #x = K.permute_dimensions(x, [0, 2, 1, 3])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = tf.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = tf.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        #x = K.permute_dimensions(x, [0, 2, 1, 3])
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    def build(self, input_shape):
        #assert len(input_shape[0]) >= 2
        print("input shape: ",input_shape)
        super(MultiLevel_BlockAttention, self).build(input_shape)
        self.return_attention = self.return_attention
        self.seq_embedding = input_shape[-1]
        self.seq_length = input_shape[1]
        self.embedding = input_shape[-1]

        self.Wq_ld = self.add_weight(name='Annotation_embedding_q_weights', shape=(self.seq_embedding, self.seq_embedding),
                                  initializer='normal', trainable=True)

        self.Wk_ld = self.add_weight(name='Annotation_embedding_k_weights', shape=(self.seq_embedding, self.seq_embedding),
                                    initializer='normal', trainable=True)

        self.Wv_ld = self.add_weight(name='Annotation_embedding_v_weights', shape=(self.seq_embedding, self.seq_embedding),
                                    initializer='normal', trainable=True)
        if self.epi_genomic:
            self.Wepigenome = self.add_weight(name='Epigenome_embedding_weights', shape=(self.seq_length, self.seq_length),
                                    initializer='normal', trainable=True)
        
        if self.use_bias:
            self.bq = self.add_weight(name='query_bias', shape=(self.seq_length,1),
                                  initializer='normal', trainable=True)
            self.bk = self.add_weight(name='key_bias', shape=(self.seq_length,1),
                                      initializer='normal', trainable=True)
            self.bv = self.add_weight(name='value_bias', shape=(self.seq_length,1),
                                      initializer='normal', trainable=True)
            self.bo = self.add_weight(name='output_bias', shape=(self.seq_length,1),
                                      initializer='normal', trainable=True)
        self.built = True
        

    def call(self, x):
        # x could be a list, while x[0] is the qkv, x[1] is the guiding attention score
        #query = tf.einsum('bsd,dd->bsd', x, self.q_ld)
        #key = tf.einsum('bsd,dd->bsd', x, self.k_ld)
        #value = tf.einsum('bsd,dd->bsd', x, self.v_ld)
        if isinstance(x,list) and len(x) >= 2:
            X,residual_score,attention_guide = x
            quit()
        else:
            X = x
            residual_score = None
            attention_guide = None
        query = X
        key = X
        value = X
        query = tf.matmul(query,self.Wq_ld)
        #tf.einsum('sd,dd->sd',X,self.wq)
        key = tf.matmul(key,self.Wk_ld)
        #tf.einsum('sd,dd->sd',X,self.wk)
        value = tf.matmul(value,self.Wv_ld)
        #tf.einsum('sd,dd->sd',X,self.wv)
        #value = tf.tensordot(x, self.wv, axes=(-1,0))
        diag_mask = tf.subtract(1.0,tf.eye(self.seq_length,dtype=tf.float32))

        if self.use_bias:
            query += self.bq
            key += self.bk
            value += self.bv
        # q,k,v shape == (batch_size, seq_len, d_model)
        if self.head_num > 1:

            query = self._reshape_to_batches(query,self.head_num)
            key = self._reshape_to_batches(key,self.head_num)
            value = self._reshape_to_batches(value,self.head_num)

            #query = tf.stack(tf.split(split_q, self.head_num, axis=2),axis=1)
            #key = tf.stack(tf.split(split_k, self.head_num, axis=2),axis=1)
            #value = tf.stack(tf.split(split_v, self.head_num, axis=2),axis=1)
        attention = tf.matmul(query, key, transpose_b=True)/tf.math.sqrt(tf.cast(self.seq_length,dtype=tf.float32))
        #overall_score = tf.add([inner_product,self.b])
        
        if residual_score is not None:
            # add attention score with residual score
            attention = tf.add(attention, residual_score)
        #attention = tf.multiply(attention, self.Wepigenome)
        
        attention = tf.nn.softmax(attention)

        # multiply last two dimensions with Wepigenome
        if self.epi_genomic:
            #attention = tf.multiply(attention, diag_mask)
            attention = tf.multiply(attention, self.Wepigenome)
            #attention = tf.add(attention, tf.eye(self.seq_length,dtype=tf.float32))
        #Check multiply outcome
        effect = tf.matmul(attention, value)
        #effect = tf.add(epi_effect,value)
        #effect = tf.nn.relu(effect)

        if self.head_num > 1:
            effect = self._reshape_from_batches(effect,self.head_num)
            #effect = tf.squeeze(all_effects, axis=1) # (batch_size, seq_len, d_model)
        if self.use_bias:
            effect += self.bo

        if self.return_attention:
            return effect,attention

        return effect 

    def compute_output_shape(self, input_shape):
        super(MultiLevel_BlockAttention, self).compute_output_shape(input_shape)
        print(input_shape)
        print("From compute output shape")
        snp_len = input_shape[1]
        assert snp_len == self.seq_length
        if self.return_attention:
            attention_shape = (input_shape[0], input_shape[1], input_shape[1])
            return [input_shape[0]+(self.seq_length,self.embedding), attention_shape]
        return input_shape[0]+(self.seq_length,self.embedding)

    def get_config(self):
        config = super(MultiLevel_BlockAttention, self).get_config()
        config = {'annotation': self.annotation,
                  'num_heads': self.head_num,
                  'use_bias': self.use_bias,
                  'epi_genomic': self.epi_genomic,
                  'return_attention': self.return_attention}
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
class GroupedLocallyConnectedLayer(layers.Layer):
    def __init__(self, kernel_para,pos,index=0,**kwargs):
        super(GroupedLocallyConnectedLayer, self).__init__(**kwargs)
        #self.num_groups = len(reference)
        self.kernel_para = kernel_para
        self.pos = pos
        self.index = index
        #self.group_reference = reference ## reference format: [[0,1,2],[3,4,5],[6,7,8]...]

    def build(self, input_shape):
        #input_dim = input_shape[-1]
        channels,seq_length,input_dim = self.kernel_para
        self.kernel = self.add_weight(shape=(seq_length, input_dim, channels), 
                                        initializer='random_normal', 
                                        name='kernel_{}'.format(self.index)) #kernel shape (channel,seq,dim)
        super(GroupedLocallyConnectedLayer, self).build(input_shape)
     
        
    def call(self, inputs):
        #groups = []
        """
        #for i,pos in enumerate(self.group_reference):
        for i,pos in tf.data.Dataset.from_tensor_slices((indices, self.group_reference)):
            pos = pos.numpy().tolist()
            selected_input = tf.gather(inputs, pos, axis=1)
            group_cal = tf.matmul(selected_input, self.kernels[i],transpose_b=True)
            groups.append(group_cal)
        """
        selected_input = tf.gather(inputs, self.pos, axis=1)

        output = tf.keras.backend.conv1d(selected_input, self.kernel,strides=1,padding='valid')
        return output

    def compute_output_shape(self, input_shape):
        output_features = len(self.pos)
        return (input_shape[0], output_features, self.kernel_para[0])
    
    def get_config(self):
        config = super(GroupedLocallyConnectedLayer, self).get_config()
        config.update({"kernel_para": self.kernel_para,"pos":self.pos,"index":self.index})
        return config
    


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


#fully connected layers block with residual connection, downsample and batch normalization (only for relu)
def fullyConnectted_block(input, width,depth=1, activation='relu',addNorm=False,use_bias=True):

    #activation_function = layers.Activation(activation=activation)
    for i in range(depth):
        X = layers.Dense(width,activation=activation,use_bias=use_bias)(input)
        if addNorm & i != 0:
            X = add_normalization(X,input,addNorm,activation)
            #out = activation(out)
            input = X
        else:
            input = X
    return X



def residual_fl_block(input, width, activation='relu',downsample=False):
    #residual fully connected layers block
    activation_function = layers.Activation(activation=activation)
    x = layers.Dense(width,activation=activation)(input)
    if activation == 'relu':
        x = layers.BatchNormalization()(x)

    if downsample:
        input_x = input
        if input.shape[-1] != x.shape[-1]:
            filter_n = x.shape[-1]
            input_x = layers.Dense(filter_n,activation=activation)(x)
        out = layers.Add()([x, input_x])
        #out = activation(out)
        return activation_function(out)
    else:
        x = activation_function(x)
        return x

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
