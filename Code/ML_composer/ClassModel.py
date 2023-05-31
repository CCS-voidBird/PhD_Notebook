from tensorflow import keras
from keras import layers
from keras.layers import Layer
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
class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr(self.model.optimizer.iterations)
        print(f"Learning rate: {lr:.6f}")

lr_logger = LearningRateLogger()

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
        x = Conv1D(self.channels_in, self.kernel, padding="same")(first_layer)
        x = layers.Activation("relu")(x)
        x = Conv1D(self.channels_in, self.kernel, padding="same")(x)
        residual = layers.Add()([x, first_layer])
        x = layers.Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


####################

class NN:

    def __init__(self,args):
        self.name = "NN"
        self.args = args
        self.lr = args.lr
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.lr,decay_steps=10000//args.batch,decay_rate=0.9,staircase=True)
        self.optimizers = {"rmsprop": keras.optimizers.RMSprop,
                      "Adam": keras.optimizers.Adam,
                      "SGD": keras.optimizers.SGD}

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,phenos,anno=None,pheno_standard = False):

        print("USE Attention CNN MODEL as training method")
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

    def model(self, input_shape, args, optimizer="rmsprop", lr=0.00001):
        pass

    def modelCompile(self,model, optimizer="rmsprop"):
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

class Transformer(NN):
    
    #############################
    #Need work!!!!!!!!!!#
    ###########################

    def __init__(self,args):
        super(Transformer,self).__init__(args)
        self.name = "Transformer"
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

        #model.add(layers.Input(shape=input_shape, dtype="float32"))
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
        geno = decoding(geno)
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
        #model = Sequential()
        """
        Convolutional Layers
        """
        #model.add(layers.Input(shape=input_shape, dtype="float32"))
        #model.add(layers.BatchNormalization())
        input = layers.Input(shape=input_shape)
        input1 = layers.Conv1D(32, 7, strides=3, padding="same")(input)
        input1 = layers.Activation("elu")(input1)
        input1 = layers.MaxPooling1D(3)(input1)
        #input1 = layers.BatchNormalization()(input)
        input1 = layers.Conv1D(32,1,padding="same")(input1)
        input1 = layers.Activation("relu")(input1)

        X = layers.Conv1D(32, 3,padding="same")(input1)
        X = layers.LeakyReLU(alpha=0.3)(X)
        X = layers.BatchNormalization()(X)
        #first_layer = layers.Activation("linear", trainable=False)(X)
        X = layers.Conv1D(32, 3, padding="same")(X)
        X = layers.LeakyReLU(alpha=0.3)(X)
        X = layers.BatchNormalization()(X)
        #X = layers.BatchNormalization()(X)

        residual1 = layers.Add()([X, input1])
        #X = layers.LeakyReLU(alpha=0.3)(residual1)

        input2 = layers.MaxPooling1D(3)(residual1)
        input2 = layers.Conv1D(64, 1, activation="relu")(input2)
        X = layers.Conv1D(64, 3, padding="same")(input2)
        X = layers.LeakyReLU(alpha=0.3)(X)
        #X = layers.BatchNormalization()(X)

        #second_layer = layers.Activation("linear", trainable=False)(X)

        X = layers.Conv1D(64, 3, padding="same")(X)
        X = layers.LeakyReLU(alpha=0.3)(X)
        #X = layers.BatchNormalization()(X)

        residual2 = layers.Add()([X, input2])
        #X = layers.LeakyReLU(alpha=0.3)(residual)
        #X = Residual(32, 3)(X)
        #X = Residual(32, 3)(X)
        #X = Residual(32, 3)(X)
        #X = layers.BatchNormalization()(X)
        X = layers.MaxPooling1D(2)(residual2)
        X = layers.Flatten()(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(args.width)(X)
        X = layers.Activation("relu")(X)
        #X = layers.Dropout(0.2)(X)
        output = layers.Dense(1, activation="linear")(X)
        #model.add(Conv1D(32, kernel_size=5, strides=3, padding='same',input_shape=input_shape))
        #model.add(layers.Activation('relu'))
        #model.add(Residual(32, 3))
        #model.add(Residual(32, 3))
        #model.add(Residual(32, 3))
        #model.add(layers.BatchNormalization())
        #model.add(Conv1D(128, kernel_size=3, strides=3, padding='same', activation='relu'))
        #model.add(MaxPooling1D(pool_size=2))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        #model.add(Flatten())
        #model.add(Dropout(rate=0.2))
        #model.add(layers.BatchNormalization())
        # Full connected layers, classic multilayer perceptron (MLP)
        #for i in range(args.depth): model.add(Dense(args.width, activation="relu"))
        #model.add(Dense(512))
        #model.add(Dropout(0.2))
        #model.add(layers.Activation("linear"))  # The output layer uses a linear function to predict traits.
        model = keras.Model(inputs=input, outputs=output,name="DCNN")
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

        model.compile(optimizer=optimizers[optimizer], loss=self.args.loss)

        """
        Optimizers: Adam, RMSProp, SGD 
        """

        return model


class DoubleCNN(NN):
    """
    double CNN, esitmate additive SNP alleles and heterozygous SNP alleles
    """
    def __init__(self,args):
        super(DoubleCNN, self).__init__(args)
        self.name = "Double channel residual CNN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__


    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE Duo (Double) CNN MODEL as training method")
        #geno = decoding(geno)
        #geno = np.expand_dims(geno, axis=2)
        geno1 = geno
        geno1 = decoding(geno1)
        geno1 = np.expand_dims(geno1, axis=2)
        geno2 = geno.mask(geno != 1,0)
        geno2 = decoding(geno2)
        geno2 = np.expand_dims(geno2, axis=2)
        #geno = np.stack((geno1,geno2),axis=2)
        #geno = np.expand_dims(geno, axis=3)
        print("The transformed SNP shape:", geno1.shape,geno2.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return [geno1,geno2],pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.0001,annotation=None):

        input1 = layers.Input(shape=input_shape)
        X1 = layers.Conv1D(64, kernel_size=25, strides=3, padding='same')(input1)
        X1 = layers.Activation('relu')(X1)
        X1 = layers.MaxPooling1D(pool_size=2)(X1)
        X1 = layers.Conv1D(128, kernel_size=3, strides=3, padding='same')(X1)
        X1 = layers.Activation('relu')(X1)
        X1 = layers.MaxPooling1D(pool_size=2)(X1)

        input2 = layers.Input(shape=input_shape)
        X2 = layers.Conv1D(64, kernel_size=25, strides=3, padding='same')(input2)
        X2 = layers.Activation('relu')(X2)
        X2 = layers.MaxPooling1D(pool_size=2)(X2)
        X2 = layers.Conv1D(128, kernel_size=3, strides=3, padding='same')(X2)
        X2 = layers.Activation('relu')(X2)
        X2 = layers.MaxPooling1D(pool_size=2)(X2)

        X = layers.concatenate([X1, X2], axis=-1)
        X = layers.Flatten()(X)
        X = layers.Dropout(rate=0.2)(X)

        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width, downsample=(i % 2 != 0 & self.args.residual))

        output = layers.Dense(1, activation="linear")(X)
        model = keras.Model(inputs=[input1,input2], outputs=output)

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

class NCNN(NN):

    def __init__(self,args):
        super(NCNN,self).__init__(args)
        self.name = "Numeric CNN"
        self.args = args


    def model_name(self):
        #get class name
        if self.args.residual is True:
            return "Res"+self.__class__.__name__
        else:
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

        """
        Convolutional Layers
        model = Sequential()
        model.add(Conv1D(64, kernel_size=5, strides=3, padding='valid', activation='elu',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(128, kernel_size=3, strides=3, padding='valid', activation='elu'))
        model.add(MaxPooling1D(pool_size=2))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        model.add(Dropout(rate=0.2))

        model.add(Flatten())

        # Full connected layers, classic multilayer perceptron (MLP)
        for i in range(args.depth):
            model.add(Dense(args.width, activation="elu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
        """
        input = layers.Input(shape=input_shape)
        X = layers.Conv1D(64, kernel_size=50, strides=10, padding='same', activation='relu',use_bias=False)(input)
        X = layers.BatchNormalization()(X)
        X = layers.MaxPooling1D(pool_size=5)(X)
       

        X = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='elu')(X)
        X = layers.MaxPooling1D(pool_size=2)(X)
        print(X.shape)
        #X = layers.MaxPooling1D(pool_size=2)(X)
        X = layers.Dense(128,activation='linear',use_bias=False)(X)
        X = layers.Dense(5,activation='linear')(X)
        X = layers.Flatten()(X)
        '''
        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width,activation=layers.ELU(),downsample=(i%2 != 0 & self.args.residual))
        '''
        X = layers.Dropout(rate=0.2)(X)
        X1 = fullyConnectted_block(X, args.width, args.depth,activation=args.activation,use_bias=False)
        non_linear_output = layers.Dense(1, activation="linear")(X1)
        linear_output = layers.Dense(1, activation="linear")(X)
        output = layers.Add()([non_linear_output, linear_output])

        model = keras.Model(inputs=input, outputs=output)

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

        model.compile(optimizer=optimizers[optimizer], loss=self.args.loss)

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


class MLP():

    def __init__(self,args):
        self.args = args
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

class Double_MLP():

    def __init__(self,args):
        self.args = args
        self.name = "Double_MLP"

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

        input1 = layers.Input(shape=input_shape)
        X = fullyConnectted_block(input1, args.width, args.depth,activation=args.activation)
        non_linear_output = layers.Dense(1, activation="linear")(X)
        linear_output = layers.Dense(1, activation="linear")(input1)
        output = layers.Add()([non_linear_output, linear_output])

        model = keras.Model(inputs=input1, outputs=output)

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

        model.compile(optimizer=optimizers[optimizer], loss=self.args.loss)

        return model

class ResMLP(NN):

    def __init__(self,args):
        super(ResMLP,self).__init__(args)
        self.name = "Residual MLP"
        self.args = args

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

        # model.add(Dropout(0.2))

        input = layers.Input(shape=input_shape)
        X = layers.BatchNormalization()(input)

        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width,activation=layers.ELU(),downsample=(i % 2 != 0 & self.args.residual))

        output = layers.Dense(1, activation="linear")(X)

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

        model = keras.Model(inputs=input, outputs=output)

        model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error")

        return model

class LNN(NN):

    def __init__(self,args):
        super(LNN,self).__init__(args)
        self.name = "Local NN"
        self.args = args


    def model_name(self):
        #get class name
        if self.args.residual is True:
            return "Res"+self.__class__.__name__
        else:
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

        input = layers.Input(shape=input_shape)
        X = layers.ZeroPadding1D(padding=(0, input_shape[1] // 5))(input)
        X = layers.LocallyConnected1D(args.locallyConnect, kernel_size=5, strides=5, padding='valid', activation='elu')(X) # recommend to have 128 LNN channels
        X = layers.Conv1D(256, kernel_size=20, strides=2, padding='valid', activation='elu')(X)

        #X = layers.LocallyConnected1D(128, kernel_size=3, strides=3, padding='valid', activation='elu')(X)

        #X = layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(X)
        X = layers.Flatten()(X)
        X = layers.Dropout(0.2)(X)
        for i in range(args.depth):
            X = residual_fl_block(input=X, width=self.args.width,activation=layers.ELU(),downsample=(i%2 != 0 & self.args.residual))


        output1 = layers.Dense(1, activation="linear")(X)
        output2 = fullyConnectted_block(X, args.width, args.depth, "sigmoid")
        output2 = layers.Dense(1, activation="linear")(output2)
        output = layers.Add()([output1,output2])
        model = keras.Model(inputs=input, outputs=output)

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

        model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error",metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])

        """
        Optimizers: Adam, RMSProp, SGD 
        """

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

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001):
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

class MultiHeadAttentionLNN(NN):
    """
    Multi Head Attention with RealFormer (Residual Transformer) structure
    """

    def __init__(self,args):
        super(MultiHeadAttentionLNN,self).__init__(args)
        self.name = "LocalRealFormer"
        self.rank = True  ##rank block value to 0 (zero),1 (low),2 (high).
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,pheno,anno=None,pheno_standard = False):

        print("USE LocalRealFormer MODEL as training method")
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        geno = tf.convert_to_tensor(geno)
        geno = tf.cast(geno,dtype=tf.float32)
        print("The transformed SNP shape:", geno.shape)

        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        pheno = tf.convert_to_tensor(pheno)
        pheno = tf.cast(pheno, dtype=tf.float32)
        print("The transformed SNP shape:", pheno.shape,pheno.dtype)
        return geno,pheno

    def model(self, input_shape,args, optimizer="rmsprop", lr=0.00001,annotation=None):
        # init Q,K,V
        depth = args.depth
        embed = args.embedding
        activation = args.activation
        input1 = layers.Input(shape=input_shape,name="input_layer_1")
        print(input1.shape)

        if annotation is None:

            X = layers.ZeroPadding1D(padding=(0, input_shape[1] // 10))(input1)

            V = layers.LocallyConnected1D(args.embedding, 10, strides=10, activation=activation, padding="valid",
                                          use_bias=False)(X)

        else:

            V = SNPBlockLayer(channels=args.embedding)(input1,annotation)

        V = layers.Dense(embed,activation=activation)(V)

        M1,AM1 = MultiHead_QKV_BlockAttention(args.num_heads,residual=None)([V])
        M1 = layers.Add()([M1,V])
        M2 = layers.LayerNormalization()(M1)
        M = residual_fl_block(input=M2, width=embed,activation=activation, downsample=True)
        #M2 = residual_fl_block(input=M1, width=self.args.width, downsample=True)
        #M = layers.Dropout(0.4)(M)
        M3,AM3 = MultiHead_QKV_BlockAttention(args.num_heads,residual=True)([M, AM1])
        M3 = layers.Add()([M3,M])
        M3 = layers.LayerNormalization()(M3)
        M3 = residual_fl_block(input=M3,activation=activation, width=embed, downsample=True)
        M = layers.Flatten()(M3)
        # M = tf.reduce_sum(M3,axis=1)
        M = layers.Dropout(0.2)(M)

        if self.args.data_type == "ordinal":

            while depth > 0:
                M = residual_fl_block(input=M, width=self.args.width, activation=activation,
                                      downsample=(depth % 2 == 0 & self.args.residual))
                depth -= 1
            QV_output = OrdinalOutputLayer(num_classes=self.args.classes)(M)
            loss_class = Ordinal_loss(self.args.classes)
        else:
            while depth > 0:
                M = residual_fl_block(input=M, width=self.args.width, activation=activation,
                                      downsample=(depth % 2 == 0 & self.args.residual))
                depth -= 1
            QV_output = layers.Dense(1, activation="linear")(M)

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

        model = keras.Model(inputs=input1, outputs=QV_output)

        if self.args.data_type == "ordinal":
            model.compile(optimizer=optimizers[optimizer], loss=loss_class.loss,metrics=['acc'])
        else:
            model.compile(optimizer=optimizers[optimizer], loss="mean_squared_error",metrics=['acc'])

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

        print("USE Attention CNN MODEL as training method")
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

    def model(self, input_shape, args, optimizer="rmsprop", lr=0.001, annotation=None):
        # init Q,K,V
        depth = args.depth
        embed = args.embedding
        activation = args.activation
        input1 = layers.Input(shape=input_shape, name="input_layer_1")
        print(input1.shape)
        zero_padding = args.locallyBlock - input_shape[0] % args.locallyBlock

        if annotation is None:

            X = layers.ZeroPadding1D(padding=(0, zero_padding))(input1)

            V = layers.LocallyConnected1D(args.locallyConnect, args.locallyBlock, strides=args.locallyBlock, activation=activation, padding="valid",
                                          use_bias=False)(X)

        else:
            groups_sizes = [len(x) for x in annotation]
            #V = GroupedLocallyConnectedLayer(channels=args.embedding,reference=annotation)(input1)

            kernel_paras = [(args.embedding,groups_sizes[i],input_shape[-1]) for i in range(len(annotation))]
            Xs = [GroupedLocallyConnectedLayer(kernel_para,annotation[index],index)(input1) for index,kernel_para in enumerate(kernel_paras)]
            V = layers.Concatenate(axis=1)(Xs)

        V = layers.Dense(embed, activation=activation)(V)

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
        M1 = MultiLevel_BlockAttention(args.num_heads, return_attention=False)(V)
        M = layers.Dense(embed, activation=activation)(M1)
        #M = layers.Add()([M1, V])
        #M = layers.Flatten()(M)
        
        """
        if self.args.data_type == "ordinal":

            while depth > 0:
                M = residual_fl_block(input=M1, width=self.args.width, activation=activation,
                                      downsample=(depth % 2 == 0 & self.args.residual))
                depth -= 1
            QV_output = OrdinalOutputLayer(num_classes=self.args.classes)(M)
        else:
            print(M1.shape)
            print("From model")
            M1 = layers.Dense(self.args.embedding, activation="linear")(M1)
            M = layers.Conv1D(1, kernel_size=1, strides=1,padding="same", use_bias=False)(M1)
            #X = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='elu')(X)
            D = layers.Activation("sigmoid")(M)
            D = layers.Flatten()(D)
            D = layers.Dense(1, activation="linear")(D)

            M = layers.GlobalAveragePooling1D()(M)
            M = layers.Flatten()(M)
            QV_output = layers.Add()([M, D])
        """
        M = layers.Conv1D(1, kernel_size=1, strides=1,padding="same", use_bias=False)(M1)
        #X = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='elu')(X)
        D = layers.Activation("sigmoid")(M)
        D = layers.Flatten()(D)
        D = layers.Dense(1, activation="linear")(D)

        M = layers.GlobalAveragePooling1D()(M)
        M = layers.Flatten()(M)
        #M = layers.Dense(1, activation="linear")(M) ##Only for debugging, need remove
        QV_output = layers.Add()([M, D])

        model = keras.Model(inputs=input1, outputs=[QV_output])
        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=['acc'])
        else:

            model.compile(optimizer=self.optimizers[optimizer](learning_rate=self.lr_schedule), loss=self.args.loss, metrics=['acc'])
            #model.compile(optimizer=optimizers[optimizer], loss=self.args.loss, metrics=['acc'])

        # QK = layers.Dot(axes=[2, 2])([Q_encoding, K_encoding])
        # QK = layers.Softmax(axis=-1)(QK)
        # QKV = layers.Dot(axes=[2, 1])([QK, V_encoding])
        # QKV = layers.Flatten()(QKV)
        # QKV = layers.Dense(1, activation="linear")(QKV)

        return model

class MultiLevelNN(NN):
    def __init__(self,args):
        super(MultiLevelNN,self).__init__(args)
        self.name = "MultiLevel NN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def model(self, input_shape, args, optimizer="rmsprop", lr=0.00001):

        input = layers.Input(input_shape)
        X = layers.Embedding(input_dim=3,output_dim=5,)(input)
        #M1, AM1 = layers.MultiHeadAttention(num_heads=1, key_dim=8,return_attention_scores=True)([X,X,X])
        M1, AM1 = MultiHead_QKV_BlockAttention(args.num_heads, residual=None)([X])
        M1 = layers.Add()([M1,X])
        M1 = layers.LayerNormalization()(M1)

        #M2, AM2 = layers.MultiHeadAttention(num_heads=1, key_dim=8,return_attention_scores=True)([X,X,X])
        M2, AM2 = MultiHead_QKV_BlockAttention(args.num_heads, residual=None)([M1,AM1])
        M2 = layers.Add()([M2, M1])
        M2 = layers.LayerNormalization()(M2)
        X = layers.Dense(1024, activation='relu')(M2)
        X = layers.Dropout(0.4)(X)

        X = layers.Dense(1,activation='linear')(X)
        X = layers.Flatten()(X)
        output = layers.Dense(1,activation="linear")(X)
        model = keras.Model(inputs=input, outputs=output)
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

        model.compile(optimizer=optimizers[optimizer], loss="mean_absolute_error", #mean_squared_error
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

        return model


MODELS = {
    "MLP": Double_MLP,
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
