try:
    import keras
    from keras.models import Sequential
    from tensorflow.keras import layers
    from keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D
    from keras.layers import Dropout
    from tensorflow.keras.utils import to_categorical
    import tensorflow as tf
    import keras.metrics
except:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D, Dropout
        import tensorflow as tf
        print("Use tensorflow backend keras module")
    except:
        print("This is not a GPU env.")
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import numpy as np
import configparser
from Functions import *
from tensorflow.keras.layers import Layer

# Define the residual block as a new layer
class Residual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

    def call(self, x):
        # the residual block using Keras functional API
        first_layer = layers.Activation("linear", trainable=False)(x)
        x = Conv1D(self.channels_in, self.kernel, padding="same")(first_layer)
        x = layers.Activation("relu")(x)
        x = Conv1D(self.channels_in, self.kernel, padding="same")(x)
        residual = layers.Add()([x, first_layer])
        x = layers.Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


####################
"""
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

####################
"""

class NN:

    def __init__(self,args):
        self.name = "NN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def model(self, input_shape, args, optimizer="rmsprop", lr=0.00001):
        pass

class TNN(NN):
    
    #############################
    #Need work!!!!!!!!!!#
    ###########################

    def __init__(self,args):
        self.name = "Transformer model"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        geno.replace(0.01, 0, inplace=True)
        #geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        output_dim=4
        lr = float(lr)
        model = Sequential()
        # Add an Embedding layer expecting input vocab of size sequence length, and
        # output embedding dimension of size 64.
        model.add(layers.Input(shape=input_shape, dtype="float32"))


        #model.add(layers.Embedding(input_dim=3, output_dim=output_dim))

        # Add a LSTM layer with 128 internal units.
        model.add(layers.Bidirectional(layers.LSTM(64)))

        # Add a Dense layer with defined units.
        model.add(layers.Dense(args.width))

        model.add(Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
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
        Optimizers: Adam, RMSProp, SGD 
        """

        return model


class RNN(NN):

    #############################
    # Need work!!!!!!!!!!#
    ###########################
    # super init function with RNN
    def __init__(self,args):
        NN.__init__(self,args)
        self.name = "RNN"

    def model_name(self):
        # get class name
        return self.__class__.__name__

    def data_transform(self, geno, pheno, anno=None, pheno_standard=False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        geno.replace(0.01, 0, inplace=True)
        # geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno, pheno

    def model(self, input_shape, args, optimizer="rmsprop", lr=0.00001):
        embed_dim = 32  # Embedding size for each token
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        output_dim = 4
        lr = float(lr)
        model = Sequential()
        # Add an Embedding layer expecting input vocab of size sequence length, and
        # output embedding dimension of size 64.
        model.add(layers.Input(shape=input_shape, dtype="float32"))

        # model.add(layers.Embedding(input_dim=3, output_dim=output_dim))

        # Add a LSTM layer with 128 internal units.
        model.add(layers.Bidirectional(layers.LSTM(64)))

        # Add a Dense layer with defined units.
        model.add(layers.Dense(args.width))

        model.add(Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
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
        Optimizers: Adam, RMSProp, SGD 
        """

        return model


class DCNN():
    """
    double CNN, esitmate additive SNP alleles and heterozygous SNP alleles
    """
    def __init__(self):
        self.name = "Deep Residual CNN"

    def model_name(self):
        #get class name
        return self.__class__.__name__


    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Deep Residual CNN MODEL as training method")
        geno1 = geno
        geno2 = geno.mask(geno != 1,0)
        #overlap geno1 and geno2 to one matrix
        #geno1 = np.expand_dims(geno1, axis=2)
        #geno2 = np.expand_dims(geno2, axis=2)
        geno = np.stack((geno1,geno2),axis=2)

        print("The transformed SNP shape:", geno.shape)
        print(geno[0:10,0:10,1])
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        lr = float(lr)
        model = Sequential()
        """
        Convolutional Layers
        """
        #model.add(layers.Input(shape=input_shape, dtype="float32"))
        #model.add(layers.BatchNormalization())
        model.add(Conv1D(32, kernel_size=5, strides=3, padding='same',input_shape=input_shape))
        model.add(layers.Activation('relu'))
        model.add(Residual(32, 3))
        model.add(Residual(32, 3))
        model.add(Residual(32, 3))
        #model.add(layers.BatchNormalization())
        #model.add(Conv1D(128, kernel_size=3, strides=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        model.add(Flatten())
        model.add(Dropout(rate=0.2))
        #model.add(layers.BatchNormalization())
        # Full connected layers, classic multilayer perceptron (MLP)
        #for i in range(args.depth): model.add(Dense(args.width, activation="relu"))
        model.add(Dense(512))
        model.add(Dropout(0.2))
        model.add(layers.Activation("linear"))  # The output layer uses a linear function to predict traits.
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
        Optimizers: Adam, RMSProp, SGD 
        """

        return model

class NDCNN():
    """
    double CNN, esitmate additive SNP alleles and heterozygous SNP alleles
    """
    def __init__(self):
        self.name = "ND CNN"

    def model_name(self):
        #get class name
        return self.__class__.__name__


    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Duo (Double) CNN MODEL as training method")
        geno1 = geno
        geno2 = geno.mask(geno != 1,0)
        geno = np.stack((geno1,geno2),axis=2)
        geno = np.expand_dims(geno, axis=3)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        lr = float(lr)
        args = args
        model = Sequential()
        """
        Convolutional Layers
        """
        model.add(Conv2D(16, kernel_size=[1,2], strides=[1,1], padding='valid', activation='relu',
                         input_shape=input_shape))

        model.add(Conv2D(64, kernel_size=[5,1], strides=[3, 1], padding='valid', activation='elu',
                         input_shape=input_shape))

        #model.add(MaxPooling2D(pool_size=[2,1]))

        model.add(Conv2D(128, kernel_size=[3,1], strides=[3,1], padding='valid', activation='elu'))
        #model.add(MaxPooling2D(pool_size=[2,1]))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        model.add(Dropout(rate=0.2))

        model.add(Flatten())
        # Full connected layers, classic multilayer perceptron (MLP)
        while args.depth > 0:
            model.add(Dense(args.width, activation="elu"))
            args.depth -= 1

        model.add(Dropout(0.2))
        model.add(Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
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
        Optimizers: Adam, RMSProp, SGD 
        """

        return model


class NCNN():

    def __init__(self):
        self.name = "Numeric CNN"

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        lr = float(lr)
        model = Sequential()
        """
        Convolutional Layers
        """
        model.add(Conv1D(64, kernel_size=5, strides=3, padding='valid', activation='elu',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(128, kernel_size=3, strides=3, padding='valid', activation='elu'))
        model.add(MaxPooling1D(pool_size=2))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        model.add(Dropout(rate=0.2))

        model.add(Flatten())

        # Full connected layers, classic multilayer perceptron (MLP)
        for layers in range(args.depth):
            model.add(Dense(args.width, activation="elu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
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
        Optimizers: Adam, RMSProp, SGD 
        """

        return model


    
    
class BCNN():

    def __init__(self):
        self.name = "Binary CNN"

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self, geno, pheno, anno=None,pheno_standard = False):
        print("USE Binary CNN MODEL as training method")
        geno = decoding(geno)
        geno.replace(0.01, 3, inplace=True)
        geno = to_categorical(geno)
        print("The transformed SNP shape:",geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
        lr = float(lr)
        model = Sequential()
        """
        Convolutional Layers
        """
        model.add(Conv1D(64, kernel_size=5, strides=3, padding='valid', activation='elu',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(128, kernel_size=3, strides=3, padding='valid', activation='elu'))
        model.add(MaxPooling1D(pool_size=2))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        model.add(Dropout(rate=0.2))

        model.add(Flatten())

        # Full connected layers, classic multilayer perceptron (MLP)
        for layers in range(args.depth):
            model.add(Dense(args.width, activation="elu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
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
        Optimizers: Adam, RMSProp, SGD 
        """

        return model

def DeepGS(input_shape,n_layers=0,n_units=32,optimizer="SGD",lr=0.01):
    print("If need to specify MLP parameters for DeepGS model, modify MODEL file instead.")
    lr = float(lr)
    model = Sequential()
    model.add(Conv1D(8, kernel_size=18, strides=1, padding='valid', activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=4,strides=4))
    model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dropout(rate=0.1))
    model.add(Dense(32,activation="relu"))
    model.add(Dropout(rate=0.05))
    model.add(Dense(1,activation="linear"))

    try:
        adm = keras.optimizers.Adam(learning_rate=lr)
        rms = keras.optimizers.RMSprop(learning_rate=lr)
        sgd = keras.optimizers.SGD(learning_rate=lr)
    except:
        adm = keras.optimizers.Adam(lr=lr)
        rms = keras.optimizers.RMSprop(lr=lr)
        sgd = keras.optimizers.SGD(lr=lr)

    optimizers = {"rmsprop":rms,
                 "Adam": adm,
                 "SGD": sgd}

    model.compile(optimizer=optimizers[optimizer],loss="mean_squared_error")

    return model

class MLP():

    def __init__(self):
        self.name = "MLP"

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Numeric CNN MODEL as training method")
        geno = decoding(geno)
        #geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
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

        return model




def RF(config = None,specific=False,n_features = 500,n_estimators = 200):
    if specific == True:
        model = RandomForestRegressor(n_jobs=-1, random_state=0, criterion="mse", oob_score=False, verbose=1,max_features=n_features,
                                      n_estimators=n_estimators)
        return model
    else:
        rm_config = {x: int(config["RM"][x]) for x in config["RM"].keys()}
        try:
            model = RandomForestRegressor(n_jobs=-1, random_state=0, criterion="mse", oob_score=False, verbose=1,
                                          **rm_config)
        except:
            print("Cannot find the config file 'MLP_parameters.ini")
            model = RandomForestRegressor(n_jobs=-1, random_state=0, criterion="mse", oob_score=False, verbose=1,
                                          n_estimators=2000)

        return model



MODELS = {
    "MLP": MLP,
    "Numeric CNN": NCNN,
    "Binary CNN": BCNN,
    "Test CNN":TNN,
    "Duo CNN": DCNN,
    "DeepGS": DeepGS,
    "ND CNN": NDCNN,
}

METHODS = {
    "MLP": MLP,
    "NCNN": NCNN,
    "BCNN": BCNN,
    "DeepGS": DeepGS
}

def main():
    print("Main function from ClassModel.py")
    #tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()
