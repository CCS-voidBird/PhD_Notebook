from tensorflow import keras
from keras import layers
from keras.layers import Layer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchsummary import summary

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import numpy as np
from Functions import *
#from tensorflow.keras.layers import Layer
from keras.callbacks import LearningRateScheduler
from CustomLayers import *
tf.config.experimental_run_functions_eagerly(True)
# Define the residual block as a new layer
'''
def step_decay(epoch):
    initial_lr = 0.001
    drop_rate = 0.5
    epochs_drop = 5
    lr = initial_lr * drop_rate ** (epoch // epochs_drop)
    return lr
'''
loss_fn = {
    "mse": "mse",
    "mae": "mae",
    "cor_mse": Cor_mse_loss().loss,
    "r2": R2_score_loss().loss,
}

act_fn = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "linear": "linear",
    "softmax": "softmax",
    "leaky_relu": layers.LeakyReLU(alpha=0.1),
    "elu": "elu"
}
class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr(self.model.optimizer.iterations)
        print(f"Learning rate: {lr:.6f}")

lr_logger = LearningRateLogger()

def p_corr(y_true, y_pred):
    pearson_correlation = tfp.stats.correlation(y_true, y_pred)
    return pearson_correlation

def r2_score(observations,predictions):
    total_error = tf.reduce_sum(tf.square(tf.subtract(observations, tf.reduce_mean(observations))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(observations, predictions)))
    R_squared = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    return R_squared


def addNormLayer(_input=None,_residual=None,switch=False,normType="batch"):

    V = layers.Dropout(0.2)(_input)
    if switch:
        if switch == "AddNorm":
            V = layers.Add()([V, _residual])
        if normType == "batch":
            V = layers.BatchNormalization()(V)
        if normType == "layer":
            V = layers.LayerNormalization()(V)
        #V = layers.Activation("relu")(V)
    return V

class Residual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel
        #self.Conv1D = layers.Conv1D(self.channels_in, self.kernel, padding="same")
        #self.Activation = layers.Activation("relu")
    def call(self, x):
        # the residual block using Keras functional API
        first_layer = layers.Activation("linear", trainable=False)(x)
        x = layers.Conv1D(self.channels_in, self.kernel, padding="same")(first_layer)
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(self.channels_in, self.kernel, padding="same")(x)
        residual = layers.Add()([x, first_layer])
        x = layers.Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
    

class NN:

    def __init__(self,args):
        self.name = "NN"
        self.args = args
        self.lr = args.lr
        decay_steps=args.numDecay//args.batch if args.numDecay else 10000
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.lr,decay_steps=decay_steps,decay_rate=0.9,staircase=True)
        self.optimizers = {"rmsprop": keras.optimizers.RMSprop,
                      "Adam": keras.optimizers.Adam,
                      "SGD": keras.optimizers.SGD}
        self.lossfunc = loss_fn[self.args.loss] #For external validation

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,phenos,anno=None,pheno_standard = False):

        print("USE {} MODEL as training method".format(self.name))
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        #pos = np.arrays(range(geno.shape[1]))
        #pos = np.expand_dims(pos, axis=0)
        print("The transformed SNP shape:", geno.shape)
        # for multiple phenos
        if isinstance(phenos, list):
            for i in range(len(phenos)):
                if pheno_standard is True:
                    phenos[i] = stats.zscore(phenos[i])
        return geno,phenos

    def model(self, input_shape, args, optimizer="Adam", lr=0.00001):
        pass

    def modelCompile(self,model, optimizer="Adam"):
        adm = keras.optimizers.Adam
        rms = keras.optimizers.RMSprop
        sgd = keras.optimizers.SGD

        optimizers = {"rmsprop": rms,
                      "Adam": adm,
                      "SGD": sgd}

        model.compile(optimizer=optimizers[optimizer](learning_rate=self.lr_schedule), loss=self.args.loss)

        """
        Optimizers: Adam, RMSProp, SGD 
        """

        return model

###################
# Torch based transformer model
###################

class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(positional_encoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.Wq(q).view(batch_size, -1, self.num_heads, self.depth)
        k = self.Wk(k).view(batch_size, -1, self.num_heads, self.depth)
        v = self.Wv(v).view(batch_size, -1, self.num_heads, self.depth)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.depth)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.Wo(output)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x1 = self.mha(x, x, x, mask)
        x = x + self.dropout1(x1)
        x1 = self.layernorm1(x)
        x1 = self.ff(x1)
        x = x + self.dropout2(x1)
        x = self.layernorm2(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.pe = positional_encoding(d_model)
    
    def forward(self, x, mask):
        x = x + self.pe(x)
        x = self.encoder(x, mask)
        return x
        
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.fc(x)
        return x
    
    def summary(self):
        # Print model summary by layers
        summary(self, (1000, 1))
        return
    
    def fit(self, x, y, epochs, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self(x, mask=None)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
    def predict(self, x):
        return self(x, mask=None)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    
class TransformerModel(NN):
    def __init__(self,args):
        super(TransformerModel,self).__init__(args)
        self.name = "Transformer"
        self.rank = True  ##rank block value to 0 (zero),1 (low),2 (high).
        self.args = args
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam
        self.epoch = args.epoch

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):

        print("USE Transformer MODEL as training method")
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        #pos = np.arrays(range(geno.shape[1]))
        #pos = np.expand_dims(pos, axis=0)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="Adam", lr=0.00001):
        model = Transformer(input_shape[1], args.num_heads, args.width, args.depth)
        return model
    
####################


class NCNN(NN):

    def __init__(self,args):
        super(NCNN,self).__init__(args)
        self.name = "Numeric CNN"
        self.args = args


    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE {} MODEL as training method".format(self.name))
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="Adam", lr=0.00001):
        lr = float(lr)
        input1 = layers.Input(shape=input_shape)
        X = layers.Conv1D(64, kernel_size=5, strides=3, padding='same', activation=act_fn[args.activation])(input1)
        #X = add_normalization(X,input1,norm_switch=False,activation=self.args.activation)
        X1 = layers.MaxPooling1D(pool_size=2,strides=1)(X)
       
        X = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation=act_fn[args.activation])(X1)
        #X = add_normalization(X,X1,norm_switch=False,activation=self.args.activation)
        X = layers.MaxPooling1D(pool_size=2,strides=1)(X)
        #X = layers.Dense(1, activation=act_fn[args.activation])(X)
        #X = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation=act_fn[args.activation])(X)
        #X = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation=act_fn[args.activation])(X)
        #X = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation=act_fn[args.activation])(X)
        #X = layers.MaxPooling1D(pool_size=3,strides=1)(X)
        #X = layers.Dropout(rate=0.2)(X)
        X = layers.Flatten()(X)
        #X = layers.Dropout(rate=0.2)(X)
        X = fullyConnectted_block(X, args.width, args.depth,activation=act_fn[self.args.activation],addNorm = self.args.addNorm, use_bias=True)
        X = tf.expand_dims(X, axis=-1)
        M = layers.Conv1D(1, kernel_size=1, strides=1,padding="same", use_bias=True,activation='linear')(X)
        GEBV = layers.GlobalAveragePooling1D()(M)
        GEBV = layers.Flatten()(GEBV)
        #M = layers.Dense(1, activation="linear")(M)
        #M = layers.Dense(1, activation="linear")(M) ##Only for debugging, need remove
        #QV_output = layers.Concatenate(axis=-1)([M, D])
        #QV_output = layers.Dense(1, activation="linear",use_bias=True)(QV_output)
        if self.args.residual is True:
            D = layers.Activation("sigmoid")(M)
            D = layers.Flatten()(D)
            D = layers.Dense(1, activation="linear")(D)
            GEBV = layers.Add()([GEBV, D])
        '''
        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width,activation=layers.ELU(),downsample=(i%2 != 0 & self.args.residual))
        '''
        #X = layers.Dropout(rate=0.2)(X)
        #X1 = fullyConnectted_block(X, args.width, args.depth,activation=act_fn[self.args.activation],use_bias=False)
        #non_linear_output = layers.Dense(1, activation="linear")(X1)
        #linear_output = layers.Dense(1, activation="linear")(X)
        #output1 = layers.Add()([non_linear_output, linear_output])

        model = keras.Model(inputs=input1, outputs=[GEBV])

        try:
            adm = keras.optimizers.Adam(learning_rate=lr)
            rms = keras.optimizers.RMSprop(learning_rate=lr)
            sgd = keras.optimizers.SGD(learning_rate=lr)
        except:
            adm = keras.optimizers.Adam(lr=lr)
            rms = keras.optimizers.RMSprop(lr=lr)
            sgd = keras.optimizers.SGD(lr=lr)

        optimizers = {"rmsprop": rms,
                      "Adam": adm,
                      "SGD": sgd}

        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=['acc'])
        else:

            model.compile(optimizer=self.optimizers[optimizer](learning_rate=self.lr_schedule), loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])

        """
        Optimizers: Adam, RMSProp, SGD 
        """

        return model
   
class MLP(NN):

    def __init__(self,args):
        super(MLP,self).__init__(args)
        self.args = args
        self.name = "MLP"

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE {} MODEL as training method".format(self.name))
        geno = decoding(geno)
        #geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="Adam", lr=0.00001):

        lr = float(lr)
        input1 = layers.Input(shape=input_shape)
        X = fullyConnectted_block(input1, args.width, args.depth,activation=act_fn[self.args.activation],addNorm = self.args.addNorm, use_bias=True)
        X = tf.expand_dims(X, axis=-1)
        M = layers.Conv1D(1, kernel_size=1, strides=1,padding="same", use_bias=False,activation='linear')(X)
        GEBV = layers.GlobalAveragePooling1D()(M)
        GEBV = layers.Flatten()(GEBV)
        #M = layers.Dense(1, activation="linear")(M)
        #M = layers.Dense(1, activation="linear")(M) ##Only for debugging, need remove
        #QV_output = layers.Concatenate(axis=-1)([M, D])
        #QV_output = layers.Dense(1, activation="linear",use_bias=True)(QV_output)
        if self.args.residual is True:
            D = layers.Activation("sigmoid")(M)
            D = layers.Flatten()(D)
            D = layers.Dense(1, activation="linear")(D)
            GEBV = layers.Add()([GEBV, D])
        '''
        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width,activation=layers.ELU(),downsample=(i%2 != 0 & self.args.residual))
        '''
        #X = layers.Dropout(rate=0.2)(X)
        #X1 = fullyConnectted_block(X, args.width, args.depth,activation=act_fn[self.args.activation],use_bias=False)
        #non_linear_output = layers.Dense(1, activation="linear")(X1)
        #linear_output = layers.Dense(1, activation="linear")(X)
        #output1 = layers.Add()([non_linear_output, linear_output])

        model = keras.Model(inputs=input1, outputs=[GEBV])

        try:
            adm = keras.optimizers.Adam(learning_rate=lr)
            rms = keras.optimizers.RMSprop(learning_rate=lr)
            sgd = keras.optimizers.SGD(learning_rate=lr)
        except:
            adm = keras.optimizers.Adam(lr=lr)
            rms = keras.optimizers.RMSprop(lr=lr)
            sgd = keras.optimizers.SGD(lr=lr)

        optimizers = {"rmsprop": rms,
                      "Adam": adm,
                      "SGD": sgd}

        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=['acc'])
        else:

            model.compile(optimizer=self.optimizers[optimizer](learning_rate=self.lr_schedule), loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])

        """       
        model = Sequential()
        model.add(Dense(args.width, activation="elu", input_shape=input_shape))
        for layers in range(args.depth - 1):
            model.add(Dense(args.width, activation="elu"))
        # model.add(Dropout(0.2))

        model.add(Dense(1, activation="linear"))

        try:
            adm = keras.optimizers.Adam(learning_rate=lr)
            rms = keras.optimizers.RMSprop(learning_rate=lr)
            sgd = keras.optimizers.SGD(learning_rate=lr)
        except:
            adm = keras.optimizers.Adam(lr=lr)
            rms = keras.optimizers.RMSprop(lr=lr)
            sgd = keras.optimizers.SGD(lr=lr)

        optimizers = {"rmsprop": rms,
                      "Adam": adm,
                      "SGD": sgd}

        model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error")
        """
        return model

class AttentionCNN(NN):

    def __init__(self,args):
        super(AttentionCNN,self).__init__(args)
        self.name = "Attention CNN"
        self.rank = True  ##rank block value to 0 (zero),1 (low),2 (high).
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):

        print("USE Attention CNN MODEL as training method")
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        #pos = np.arrays(range(geno.shape[1]))
        #pos = np.expand_dims(pos, axis=0)
        print("The transformed SNP shape:", geno.shape)

        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="Adam", lr=0.00001):
        # init Q,K,V
        if args.depth < 1:
            depth = 1
        else:
            depth = args.depth
        input1 = layers.Input(shape=input_shape,name="input_layer_1")

        X = layers.ZeroPadding1D(padding=(0, input_shape[1]//10))(input1)

        V = layers.LocallyConnected1D(1,10,strides=10, activation="relu",padding="valid",use_bias=False)(X)
        M = layers.Conv1D(8, kernel_size=1, strides=1, activation='relu', use_bias=False)(V)
        M = layers.BatchNormalization()(M)
        # V = layers.LayerNormalization()(V)
        M,value = MultiHead_Seq_BlockAttention()(M)
        M = layers.BatchNormalization()(M)
        M = layers.Dropout(0.4)(M)
        M = MultiHead_conv_BlockAttention(8)([M,value])
        #seq = M.shape[-1]
        #M = Conv2D(seq,(1,1),(1,1))(M)  #b,q,d,s -> b,n,d,s -> b,n,s

        M = layers.GlobalAvgPool2D()(M) #out shape b,s

        #M = layers.Dense(256,activation="relu")(M)
        M = layers.Dropout(0.2)(M)
        model_output = layers.Dense(1,activation='linear')(M)

        try:
            adm = keras.optimizers.Adam(learning_rate=lr)
            rms = keras.optimizers.RMSprop(learning_rate=lr)
            sgd = keras.optimizers.SGD(learning_rate=lr)
        except:
            adm = keras.optimizers.Adam(lr=lr)
            rms = keras.optimizers.RMSprop(lr=lr)
            sgd = keras.optimizers.SGD(lr=lr)

        optimizers = {"rmsprop": rms,
                      "Adam": adm,
                      "SGD": sgd}

        model = keras.Model(inputs=input1, outputs=model_output)
        model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error")

        #QK = layers.Dot(axes=[2, 2])([Q_encoding, K_encoding])
        #QK = layers.Softmax(axis=-1)(QK)
        #QKV = layers.Dot(axes=[2, 1])([QK, V_encoding])
        #QKV = layers.Flatten()(QKV)
        #QKV = layers.Dense(1, activation="linear")(QKV)

        return model

class MultiLevelAttention(NN):
    """
    Need work
    SNP (by LD mask) * LD Attention = Global averaging GEBVs
    self attention - encoding - additive
    cross attention - decoding (with raw SNP/LD map) assume to estimate global effect.
    """

    def __init__(self,args):
        super(MultiLevelAttention,self).__init__(args)
        self.name = "MultiLevelAttention"
        self.rank = False  ##rank block value to 0 (zero),1 (low),2 (high).
        self.attention_block = 2
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,phenos,anno=None,pheno_standard = False):

        print("USE MultiLevel Attention MODEL as training method")
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        #pos = np.arrays(range(geno.shape[1]))
        #pos = np.expand_dims(pos, axis=0)
        print("The transformed SNP shape:", geno.shape)
        # for multiple phenos
        if isinstance(phenos, list):
            for i in range(len(phenos)):
                if pheno_standard is True:
                    phenos[i] = stats.zscore(phenos[i])
        return geno,phenos

    def model(self, input_shape, args, optimizer="Adam", lr=0.001, annotation=None):
        # init Q,K,V
        depth = args.depth
        embed = args.embedding
        activation = args.activation
        input1 = layers.Input(shape=input_shape, name="input_layer_1")
        print(input1.shape)
        zero_padding = args.locallyBlock - input_shape[0] % args.locallyBlock

        if annotation is None:

            X = layers.ZeroPadding1D(padding=(0, zero_padding))(input1)
            V = layers.LocallyConnected1D(filters=args.locallyConnect, 
                                          kernel_size=args.locallyBlock, 
                                          strides=args.locallyBlock,
                                          activation=act_fn[self.args.activation],
                                          padding="valid",use_bias=False,
                                          kernel_regularizer=keras.regularizers.l2(0.01))(X)
            #C = layers.Conv1D(args.locallyConnect,kernel_size=10, strides=1, activation=act_fn[self.args.activation], padding="same",use_bias=False)(V)
            #V = layers.Add()([V,C])
            #Xhet = BinaryConversionLayer(condition=lambda x: x == 1.0)(X)
            #X = layers.LocallyConnected2D(filters=1,kernel_size=(1,2),strides=1,activation="relu", padding="valid")(Xhet)
            #Xhet = layers.Conv2D(filters=1,kernel_size=(1,2),strides=1,activation="relu", padding="valid",use_bias=False)(Xhet)
            #Xhet = tf.squeeze(Xhet,axis=-1)
            #X = layers.Add()([Xhet,X])
            
            ##create a sub array that only contain dominance values (1)
            #dominance_mask = K.cast(K.equal(X, 1), dtype=K.floatx())
            ##stack dominance_mask to X
            #X = K.stack([X, dominance_mask], axis=-1)
            """
            #Locally connected to reduce array size
            V = layers.LocallyConnected1D(args.locallyConnect,kernel_size=args.locallyBlock, strides=args.locallyBlock, activation=act_fn[self.args.activation], padding="valid",use_bias=False)(X)
            """

            #Conv1D to reduce array size
            #if args.locallyBlock > 2:
            #    V = layers.AveragePooling1D(pool_size=2)(V)
            #V = layers.Dropout(0.1)(V)
        else:
            groups_sizes = [len(x) for x in annotation]
            #V = GroupedLocallyConnectedLayer(channels=args.embedding,reference=annotation)(input1)

            kernel_paras = [(args.embedding,groups_sizes[i],input_shape[-1]) for i in range(len(annotation))]
            Xs = [GroupedLocallyConnectedLayer(kernel_para,annotation[index],index)(input1) for index,kernel_para in enumerate(kernel_paras)]
            V = layers.Concatenate(axis=1)(Xs)

        V = layers.Dense(embed,activation=act_fn[self.args.activation])(V)

        """
        # train and get guide attention for bin phenotypes
        bin_M1,bin_res,attention_score = MultiLevel_BlockAttention(args.num_heads,return_attention=True)([V])
        bin_M = layers.Add()([bin_M1, V])
        bin_M = layers.Flatten()(bin_M)

        bin_output1 = OrdinalOutputLayer(num_classes=self.args.classes)(bin_M)
        bin_output2 = fullyConnectted_block(input=bin_M, width=self.args.width,depth=self.args.depth, activation='sigmoid')
        bin_output2 = OrdinalOutputLayer(num_classes=self.args.classes)(bin_output2)
        bin_output = layers.Add()([bin_output1,bin_output2])
        """
        # train and get guide attention for actual phenotypes
        for attention_block in range(args.AttentionBlock):
            #V1 = MultiLevel_BlockAttention(args.num_heads, return_attention=False,epi_genomic=self.args.epistatic)(V)
            V1 = layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=embed, value_dim=embed, dropout=0.1,
                                           kernel_regularizer=keras.regularizers.l2(0.01),
                                           bias_regularizer=keras.regularizers.l2(0.01))(V,V)
            if self.args.addNorm is True:
                V1 = addNormLayer(V1,V1,switch=self.args.addNorm,normType="layer")
            #    V1 = layers.Add()([V1, V])
            #    V1 = layers.LayerNormalization()(V1)
            #    V1 = layers.Activation("relu")(V1)
            #    V1 = layers.Dropout(0.2)(V1)
            V = layers.Dense(embed,activation=act_fn[self.args.activation])(V1)
            if self.args.addNorm is True:
                V = addNormLayer(V,V,switch=self.args.addNorm,normType="batch")
            #    V = layers.Add()([V, V1])
            #    V = layers.BatchNormalization()(V)
            #    V = layers.Activation("relu")(V)
            #    V = layers.Dropout(0.2)(V)
        
        M = layers.Conv1D(1, kernel_size=1, strides=1,padding="same", use_bias=False)(V)

        #D = layers.GlobalAveragePooling1D()(M)
        D = layers.Activation("sigmoid")(M)
        D = layers.Flatten()(D)
        D = layers.Dense(1, activation="linear")(D)
        
        #D = layers.Conv1D(1, kernel_size=1, strides=1, padding="same")(D)
        #D = layers.GlobalAveragePooling1D()(D)
        #D = layers.Flatten()(D)

        M = layers.GlobalAveragePooling1D()(M)
        M = layers.Flatten()(M)

        GEBV = layers.Add()([M, D])
        #QV_output = AddingLayer_with_bias()(GEBV)

        model = keras.Model(inputs=input1, outputs=[GEBV])
        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=[p_corr])
        else:

            model.compile(optimizer=self.optimizers[optimizer](learning_rate=self.lr_schedule), loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])
            #model.compile(optimizer=optimizers[optimizer], loss=self.args.loss, metrics=['acc'])

        # QK = layers.Dot(axes=[2, 2])([Q_encoding, K_encoding])
        # QK = layers.Softmax(axis=-1)(QK)
        # QKV = layers.Dot(axes=[2, 1])([QK, V_encoding])
        # QKV = layers.Flatten()(QKV)
        # QKV = layers.Dense(1, activation="linear")(QKV)

        return model

MODELS = {
    "MLP": MLP,
    "Numeric CNN": NCNN,
    "Binary CNN": BCNN,
    "Test CNN":Transformer,
    "Duo CNN": DCNN,
    "Double CNN": DoubleCNN,
    "Attention CNN": AttentionCNN,
    "MultiHead Attention LNN": MultiHeadAttentionLNN,
    "ResMLP": ResMLP,
    "LNN": LNN,
    "MultiLevel Attention": MultiLevelAttention,
    "MultiLevelNN":MultiLevelNN
}

def main():
    print("Main function from ClassModel.py")
    #tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()
