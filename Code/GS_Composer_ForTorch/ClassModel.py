import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_core as keras
from keras import layers, activations
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import numpy as np
from Functions import *
from keras.callbacks import LearningRateScheduler
from CustomLayers import *
tf.config.experimental_run_functions_eagerly(True)
loss_fn = {
    "mse": "mse",
    "mae": "mae",
    "huber": "huber",
    "cor_mse": Cor_mse_loss().loss,
    "var_mse": Var_mse_loss().loss,
    "bounded_mse": bounded_mse_loss().loss,
    "r2": R2_score_loss().loss,
}

act_fn = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "linear": "linear",
    "softmax": "softmax",
    "leaky_relu": layers.LeakyReLU(alpha=0.1),
    "elu": "elu",
    "gelu": "gelu",
}
class LearningRateLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr(self.model.optimizer.iterations)
        print(f"Learning rate: {lr:.6f}")

lr_logger = LearningRateLogger()

def p_corr(y_true, y_pred):
    pearson_correlation = keras.ops.correlate(y_pred, y_true,mode='valid')
    #pearson_correlation = tfp.stats.correlation(y_true, y_pred)
    return pearson_correlation

def r2_score(observations,predictions):
    total_error = tf.reduce_sum(tf.square(tf.subtract(observations, tf.reduce_mean(observations))))
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(observations, predictions)))
    R_squared = tf.subtract(1.0, tf.divide(unexplained_error, total_error))
    return R_squared

def addNormLayer(_input=None,_residual=None,switch=False,normType="batch"):

    V = layers.Dropout(0.2)(_input)
    if switch:
        if switch == True:
            V = layers.Add()([V, _residual])
        if normType == "batch":
            V = layers.BatchNormalization(axis=(-1))(V)
        if normType == "layer":
            V = layers.LayerNormalization(axis=(-2,-1),epsilon=1e-6)(V)
        #V = layers.Activation("relu")(V)
    return V

def common_layers(_input=None, _residual=None,args=None,method = "Multiply"):
    M = layers.Conv1D(args.NumTrait, kernel_size=1, strides=1,padding="same",
                      )(_input)

    M = layers.Reshape((args.NumTrait,-1))(M)
    for dense in range(args.depth):
        M = layers.Dense(args.width, activation="relu")(M)

    D = HarmonicLayer()(M) if method == "Harmonic" else layers.Activation("sigmoid")(M)
    if method == "Harmonic":
        D = layers.Flatten()(D)
    elif method == "Multiply":
        D = layers.Flatten()(D)
        D = layers.Dense(args.NumTrait, activation="linear",
                         )(D)

    M = layers.Flatten()(M)
    M = layers.Dense(1, activation="linear")(M)
    

    GEBV = layers.Add()([M, D])

    return GEBV



####################

class NN:

    def __init__(self,args):
        self.name = "NN"
        self.args = args
        self.lr = args.lr
        decay_steps=args.numDecay//args.batch if args.numDecay else 10000
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.lr,decay_steps=decay_steps,decay_rate=0.9,staircase=True)
        
        self.optimizers = {"rmsprop": keras.optimizers.RMSprop(
            learning_rate=self.lr_schedule
                    ),
                      "Adam": keras.optimizers.Adam(
            learning_rate=self.args.lr
                    ),
                      "SGD": keras.optimizers.SGD(
            learning_rate=self.lr_schedule
                    )}
        
        self.lossfunc = loss_fn[self.args.loss] #For external validation

        self.early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4
        )

        # Monitor learning rate
        self.lr_monitor = keras.callbacks.LearningRateScheduler(
            schedule=self.lr_schedule,
            verbose=1
        )

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def data_transform(self,geno,phenos,anno=None,pheno_standard = False,duo=False):

        print("USE {} MODEL as training method".format(self.name))
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        if duo is True:
            ##convert another geno in which all the non-zero genotype value to 1
            geno2 = geno.mask(geno != 0,1)
            # combine two geno by channels
            geno = np.concatenate((geno,geno2),axis=2)
        #pos = np.arrays(range(geno.shape[1]))
        #pos = np.expand_dims(pos, axis=0)
        print("The transformed SNP shape:", geno.shape)
        # for multiple phenos
        self.seq_dim = geno.shape[1:]

        if isinstance(phenos, list):
            for i in range(len(phenos)):
                if pheno_standard is True:
                    phenos[i] = stats.zscore(phenos[i])
        return geno,phenos

    def model(self, input_shape, args, optimizer="Adam", lr=0.00001):
        pass

    def modelCompile(self,model, optimizer="Adam"):

        model.compile(optimizer=self.optimizers[optimizer], loss=self.args.loss)

        """
        Optimizers: Adam, RMSProp, SGD 
        """

        return model

class DNNGP(NN):
    """
    double CNN, esitmate additive SNP alleles and heterozygous SNP alleles
    """
    def __init__(self,args):
        super(DNNGP, self).__init__(args)
        self.name = "Double channel residual CNN"
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__


    """def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE {} MODEL as training method".format(self.name))
        #geno = decoding(geno)
        #geno = np.expand_dims(geno, axis=2)
        geno1 = geno
        geno1 = decoding(geno1)
        geno1 = np.expand_dims(geno1, axis=2)
        geno2 = geno.mask(geno != self.args.ploidy,0)
        geno2 = decoding(geno2)
        geno2 = np.expand_dims(geno2, axis=2)

        print("The transformed SNP shape:", geno1.shape)
        if pheno_standard is True:
            pheno = stats.zscore(pheno)
        return [geno1,geno2],pheno"""

    def model(self, input_shape,args, optimizer="Adam", lr=0.0001,annotation=None):

        input1 = layers.Input(shape=input_shape)
        V1 = layers.Conv1D(filters=64, 
                              kernel_size=4, 
                              strides=4,
                              activation=act_fn[args.activation],
                              padding="valid",use_bias=True,
                              kernel_regularizer=keras.regularizers.l2(0.01),
                              bias_regularizer=keras.regularizers.l2(0.01)
                              )(input1)
        V1 = layers.Dropout(0.2)(V1)
        V1 = layers.BatchNormalization()(V1)
        V1 = layers.Conv1D(filters=64, 
                              kernel_size=4, 
                              strides=4,
                              activation=act_fn[args.activation],
                              padding="valid",use_bias=True,
                              kernel_regularizer=keras.regularizers.l2(0.01),
                              bias_regularizer=keras.regularizers.l2(0.01)
                              )(V1)
        V1 = layers.Dropout(0.2)(V1)
        V1 = layers.Conv1D(filters=64, 
                              kernel_size=4, 
                              strides=4,
                              activation=act_fn[args.activation],
                              padding="valid",use_bias=True,
                              kernel_regularizer=keras.regularizers.l2(0.01),
                              bias_regularizer=keras.regularizers.l2(0.01)
                              )(V1)

        #mask all the input1 value != 1 to 0
        
        input2 = layers.Input(shape=input_shape)
        V2 = layers.Conv1D(filters=64, 
                              kernel_size=4, 
                              strides=4,
                              activation=act_fn[args.activation],
                              padding="valid",use_bias=True
                              )(input2)
        V2 = layers.Dropout(0.2)(V2)
        V2 = layers.BatchNormalization()(V2)
        V2 = layers.Conv1D(filters=64, 
                              kernel_size=4, 
                              strides=4,
                              activation=act_fn[args.activation],
                              padding="valid",use_bias=True
                              )(V2)
        V2 = layers.Dropout(0.2)(V2)
        V2 = layers.Conv1D(filters=64, 
                              kernel_size=4, 
                              strides=4,
                              activation=act_fn[args.activation],
                              padding="valid",use_bias=True
                              )(V2)

        X = layers.concatenate([V1, V2], axis=-1)
        
        #X = layers.Flatten()(X)

        #for i in range(args.depth):
        #    X = residual_fl_block(input=X, width=self.args.width, downsample=(i % 2 != 0 & self.args.residual))

        output = common_layers(_input=X,args=self.args)
        model = keras.Model(inputs=[input1,input2], outputs=output)


        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=[p_corr])
        else:

            model.compile(optimizer=self.optimizers[optimizer], loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])

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
        return self.__class__.__name__

    """def data_transform(self,geno,pheno,anno=None,pheno_standard = False):
        print("USE {} MODEL as training method".format(self.name))
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno"""

    def model(self, input_shape,args, optimizer="Adam", lr=0.00001):
        lr = float(lr)
        input1 = layers.Input(shape=input_shape)
        
        #X = layers.Normalization()(input1)
        X = layers.Conv1D(64, kernel_size=5, strides=3, padding='same', activation=act_fn[args.activation])(input1)
        #X = FeatureSelectionLayer()(X)
        #X = add_normalization(X,input1,norm_switch=False,activation=self.args.activation)
        X1 = layers.MaxPooling1D(pool_size=2,strides=1)(X)
       
        X = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation=act_fn[args.activation])(X1)
        #X = add_normalization(X,X1,norm_switch=False,activation=self.args.activation)
        X = layers.MaxPooling1D(pool_size=2,strides=1)(X)
        X = layers.Flatten()(X)

        X = fullyConnectted_block(X, args.width, args.depth,activation=act_fn[self.args.activation],addNorm = self.args.addNorm, use_bias=True)
        X = tf.expand_dims(X, axis=-1)
        GEBV = common_layers(_input=X,args=self.args)

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

            model.compile(optimizer=self.optimizers[optimizer], loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])

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

    """def data_transform(self, geno, pheno, anno=None,pheno_standard = False):
        print("USE {} MODEL as training method".format(self.name))
        geno = decoding(geno)
        geno.replace(0.01, 3, inplace=True)
        geno = to_categorical(geno)
        print("The transformed SNP shape:",geno.shape)
        if pheno_standard is True: 
            pheno = stats.zscore(pheno)
        return geno,pheno"""

    def model(self, input_shape,args, optimizer="Adam", lr=0.00001):
        lr = float(lr)
        model = Sequential()
        """
        Convolutional Layers
        """
        model.add(layers.Conv1D(64, kernel_size=5, strides=3, padding='valid', activation='elu',
                         input_shape=input_shape))
        model.add(layers.MaxPooling1D(pool_size=2))

        model.add(layers.Conv1D(128, kernel_size=3, strides=3, padding='valid', activation='elu'))
        model.add(layers.MaxPooling1D(pool_size=2))

        # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
        model.add(layers.Dropout(rate=0.2))

        model.add(layers.Flatten())

        # Full connected layers, classic multilayer perceptron (MLP)
        for layers in range(args.depth):
            model.add(layers.Dense(args.width, activation="elu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation="linear"))  # The output layer uses a linear function to predict traits.
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

class MLP(NN):

    def __init__(self,args):
        super(MLP,self).__init__(args)
        self.args = args
        self.name = "MLP"

    def model_name(self):
        #get class name
        return self.__class__.__name__

    def model(self, input_shape,args, optimizer="Adam", lr=0.00001):

        lr = float(lr)
        input1 = layers.Input(shape=input_shape)
        #X = layers.Normalization()(input1)
        X = fullyConnectted_block(input1, args.width, args.depth,activation=act_fn[self.args.activation],addNorm = self.args.addNorm, use_bias=True)
        X = tf.expand_dims(X, axis=-1)
        GEBV = common_layers(_input=X,args=self.args)


        model = keras.Model(inputs=input1, outputs=[GEBV])

        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=['acc'])
        else:

            model.compile(optimizer=self.optimizers[optimizer], loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])


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

    def model(self, input_shape,args, optimizer="Adam", lr=0.00001):
        lr = float(lr)

        input1 = layers.Input(shape=input_shape)
        X = layers.ZeroPadding1D(padding=(0, input_shape[1] // 5))(input1)
        X = layers.LocallyConnected1D(args.locallyConnect, kernel_size=args.locallyBlock, strides=args.locallyBlock, padding='valid', activation='elu')(X) # recommend to have 128 LNN channels

        X = fullyConnectted_block(X, args.width, args.depth,activation=act_fn[self.args.activation],use_bias=False)
        #X = tf.expand_dims(X, axis=-1)
        M = layers.Conv1D(1, kernel_size=1, strides=1,padding="same", use_bias=False,activation='linear')(X)
        GEBV = layers.GlobalAveragePooling1D()(M)
        GEBV = layers.Flatten()(GEBV)

        if self.args.residual is True:
            D = layers.Activation("sigmoid")(M)
            D = layers.Flatten()(D)
            D = layers.Dense(1, activation="linear")(D)
            GEBV = layers.Add()([GEBV, D])


        model = keras.Model(inputs=input1, outputs=[GEBV])

        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=['acc'])
        else:

            model.compile(optimizer=self.optimizers[optimizer], loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])

        return model
'''
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
'''
class AttentionCNN(NN):

    def __init__(self,args):
        super(AttentionCNN,self).__init__(args)
        self.name = "Attention CNN"
        self.rank = True  ##rank block value to 0 (zero),1 (low),2 (high).
        self.args = args
        self.seq_dim = None
        self.target_matrix_dim = 256


    def model_name(self):
        #get class name
        return self.__class__.__name__

    """def data_transform(self,geno,phenos,anno=None,pheno_standard = False,duo=False):

        print("USE {} MODEL as training method".format(self.name))
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        if duo is True:
            ##convert another geno in which all the non-zero genotype value to 1
            geno2 = geno.mask(geno != 0,1)
            # combine two geno by channels
            geno = np.concatenate((geno,geno2),axis=2)
        #pos = np.arrays(range(geno.shape[1]))
        #pos = np.expand_dims(pos, axis=0)
        print("The transformed SNP shape:", geno.shape)
        # for multiple phenos
        self.seq_dim = geno.shape[1:]

        if isinstance(phenos, list):
            for i in range(len(phenos)):
                if pheno_standard is True:
                    phenos[i] = stats.zscore(phenos[i])
        return geno,phenos"""

    def model(self, input_shape,args, optimizer="Adam", lr=0.0001):
        embed = args.embedding
        #compute the parameters for the convolutional layer to achieve the targeted matrix dimension, with steps: 4, 2, 1
        kernel_length = self.seq_dim[0]  +1 - self.target_matrix_dim
        kernel_height = self.seq_dim[1]

        # init Q,K,V
        input1 = layers.Input(shape=input_shape,name="input_layer_1")

        #X = layers.ZeroPadding1D(padding=(0, input_shape[1]//10))(input1)
        #V = layers.LocallyConnected1D(1,10,strides=10, activation="relu",padding="valid",use_bias=False)(X)

        X = layers.Conv1D(self.target_matrix_dim,
                          kernel_size=kernel_length, strides=1, 
                          activation=act_fn[args.activation])(input1)
        print("first conv1D layer: ",X.shape)

        #expened dimension to 4D tensor for conv2D
        X = tf.expand_dims(X, axis=-1) #b,d1,d2 -> b,d1,d2,1

        X = layers.Conv2D(self.args.embedding,
                            kernel_size=(kernel_height, self.target_matrix_dim), strides=(1, 1),
                            activation=act_fn[args.activation])(X)
        
        #X1 = layers.Dense(self.args.embedding,activation=act_fn[self.args.activation])(X)
        #X = addNormLayer(X1,X,switch=False,normType="batch") #layers.BatchNormalization()(X)

        # Remove the convoluted dimension
        X = tf.squeeze(X, axis=2)

        for attention_block in range(args.AttentionBlock):
            #V1 = MultiLevel_BlockAttention(args.num_heads, return_attention=False,epi_genomic=self.args.epistatic)(X)
            V1 = layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=embed, value_dim=embed, dropout=0.0,
                                           )(X,X)
            if self.args.addNorm is True:
                attention = addNormLayer(V1,X,switch=self.args.addNorm,normType="layer")
                V1 = layers.Dense(self.args.embedding,activation=act_fn[self.args.activation])(attention)    

            V = layers.Dense(self.args.embedding,
                              activation=act_fn[self.args.activation])(V1)


            if self.args.addNorm is True:
                X = addNormLayer(V,attention,switch=self.args.addNorm,normType="layer")
            #    V = layers.Add()([V, V1])
            #    V = layers.BatchNormalization()(V)
            #    V = layers.Activation("relu")(V)
            #    V = layers.Dropout(0.2)(V)
        
        GEBV = common_layers(_input=X,args=self.args,method="Multiply") #method="Harmonic"

        model = keras.Model(inputs=input1, outputs=[GEBV])
        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=[p_corr])
        else:

            model.compile(optimizer=self.optimizers[optimizer], loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])

        return model

class LCLAttention(NN):

    def __init__(self,args):
        super(LCLAttention,self).__init__(args)
        self.name = "LCLAttention"
        self.rank = False  ##rank block value to 0 (zero),1 (low),2 (high).
        self.attention_block = 2
        self.args = args

    def model_name(self):
        #get class name
        return self.__class__.__name__

    """def data_transform(self,geno,phenos,anno=None,pheno_standard = False):

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
        #convert phenos to numpy array
        #phenos = np.array(phenos)
        return geno,phenos"""

    def model(self, input_shape, args, optimizer="Adam", lr=0.001, annotation=None):
        # init Q,K,V
        input1 = layers.Input(shape=input_shape, name="input_layer_1")
        print(input1.shape)
        zero_padding = args.locallyBlock - input_shape[0] % args.locallyBlock

        if annotation is None:
            #X = layers.Normalization()(input1)
            X = layers.ZeroPadding1D(padding=(0, zero_padding))(input1)
            V = layers.LocallyConnected1D(filters=args.locallyConnect, 
                                          kernel_size=args.locallyBlock, 
                                          strides=args.locallyBlock,
                                          activation=act_fn[self.args.activation],
                                          padding="valid",use_bias=False)(X)
            
            
        else:
            groups_sizes = [len(x) for x in annotation]

            kernel_paras = [(args.embedding,groups_sizes[i],input_shape[-1]) for i in range(len(annotation))]
            Xs = [GroupedLocallyConnectedLayer(kernel_para,annotation[index],index)(input1) for index,kernel_para in enumerate(kernel_paras)]
            V = layers.Concatenate(axis=1)(Xs)

        V = layers.Dense(args.embedding,activation=act_fn[self.args.activation])(V)

        # train and get guide attention for actual phenotypes
        for attention_block in range(args.AttentionBlock):
            #V1 = MultiLevel_BlockAttention(args.num_heads, return_attention=False,epi_genomic=self.args.epistatic)(LCLout)
            V1 = layers.MultiHeadAttention(num_heads=args.num_heads, key_dim=args.embedding, value_dim=args.embedding, dropout=0.0,
                                           )(V,V)
            if self.args.addNorm is True:
                attention = addNormLayer(V1,V,switch=self.args.addNorm,normType="layer")
                V1 = layers.Dense(64,activation=act_fn[self.args.activation])(attention)    
            else:
                V = layers.Dense(args.embedding,activation=act_fn[self.args.activation])(V1)


            if self.args.addNorm is True:
                V = addNormLayer(V,attention,switch=self.args.addNorm,normType="layer")

        
        GEBV = common_layers(_input=V,args=self.args,method=self.args.method) #method="Harmonic"
        #QV_output = AddingLayer_with_bias()(GEBV)

        model = keras.Model(inputs=input1, outputs=[GEBV])
        if self.args.data_type == "ordinal":
            loss_class = Ordinal_loss(self.args.classes)
            model.compile(optimizer=self.optimizers[optimizer], loss=loss_class.loss, metrics=[p_corr])
        else:

            model.compile(optimizer=self.optimizers[optimizer], loss=loss_fn[self.args.loss], metrics=[p_corr,r2_score])


        return model


MODELS = {
    "MLP": MLP,
    "Numeric CNN": NCNN,
    "Binary CNN": BCNN,
    "DNNGP": DNNGP,
    "Attention CNN": AttentionCNN,
    "LNN": LNN,
    "LCL Attention": LCLAttention,
}

def main():
    print("Main function from ClassModel.py")
    #keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()
