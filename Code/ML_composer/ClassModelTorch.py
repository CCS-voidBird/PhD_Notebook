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


# Code to cleanly swap between Pytorch and Numpy.
# Makes PyTorch much more user friendly, but not widely used. 

#Main adjustable flag. Enables or Disable GPU optimizations
USE_CUDA = 1

def cuda(obj):
    if USE_CUDA:
        if isinstance(obj, tuple):
            return tuple(cuda(o) for o in obj)
        elif isinstance(obj, list):
            return list(cuda(o) for o in obj)
        elif hasattr(obj, 'cuda'):
            return obj.cuda()
    return obj

def tovar(*arrs, **kwargs):
    tensors = [(torch.from_numpy(a) if isinstance(a, np.ndarray) else a) for a in arrs]
    vars_ = [torch.autograd.Variable(t, **kwargs) for t in tensors]
    if USE_CUDA:
        vars_ = [v.cuda() for v in vars_]
    return vars_[0] if len(vars_) == 1 else vars_


def tonumpy(*vars_):
    arrs = [(v.data.cpu().numpy() if isinstance(v, torch.autograd.Variable) else
             v.cpu().numpy() if torch.is_tensor(v) else v) for v in vars_]
    return arrs[0] if len(arrs) == 1 else arrs

#########################
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

class zero_padding(nn.Module):
    def __init__(self, len_seq, kernel_size):
        super(zero_padding, self).__init__()
        assert type(len_seq) == int
        self.padding = len_seq % kernel_size
    
    def forward(self, x):
        # pad the second last dimension of the input tensor on the right side
        return nn.functional.pad(x, (0,0, 0,self.padding))
    
class locally_connected_layer(nn.Module):
    """
    Assuming the input shape is 3D-like 1D sequence, (batch_size, sequence_length, 1) or a blocked 1D sequence with shape (batch_size, max_block_length, n)
    """

    def __init__(self, d_model, out_channels, kernel_size, use_bias=True):
        super(locally_connected_layer, self).__init__()
        self.channels = out_channels
        self.kernel_size = (kernel_size,1)
        self.stride = kernel_size
        self.locally_connection = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride) #output shape (batch_size, out_channels*kernel_size, sequence_length//kernel_size)
        self.masks = None
        self.temp_tensor = None

        if type(d_model) == int: ##assuming the input is a entire sequence without blocking
            self.d_model = d_model
            self.LC_Weight = nn.Parameter(torch.randn(out_channels, d_model, 1))
            self.LC_Bias = nn.Parameter(torch.randn(out_channels, 1)) if use_bias else None

        elif type(d_model) == tuple: ##assuming the input is a blocked sequence, with shape (batch_size, max_block_length, n_blocks)
            self.d_model = d_model[0]
            self.LC_Weight = nn.Parameter(torch.randn(out_channels, d_model[0], d_model[1]))
            self.LC_Bias = nn.Parameter(torch.randn(out_channels, d_model[1])) if use_bias else None
            

        '''
        if block_indexes is not None:
            # create a weight mask regarding to block index, with shape (out_channels, d_model, 1)
            assert len(block_indexes) > 1 ##assuming the block_indexes is a list of indexes e.g., [[0,1,2],[3,4,5],[6,7,8]]
            num_blocks = len(block_indexes)
            max_length = max(
                [len(block) for block in block_indexes]) if max_block_length is None else max_block_length
            self.temp_tensor = torch.zeros(out_channels, d_model, max_length)
            ##create a masking tensor with shape (out_channels, d_model, num_blocks) in which values are false
            self.block_mask = torch.zeros(out_channels, d_model, num_blocks, dtype=torch.bool)
            for i, block in enumerate(block_indexes): ##for each block, set the pos of the block to True
                self.block_mask[:, block, i] = True
        '''
    
    def forward(self, x):
        ###convert x shape from (batch_size, sequence_length, n) to (batch_size, 1, sequence_length, n)
        ###convert last dimension to channel dimension
        x = x.permute(0, 2, 1).unsqueeze(1)
        ###duplicate x to the number of out_channels:: (batch_size, out_channels, sequence_length, n)
        x = x.expand(-1, self.channels, -1, -1)
        ###multiply the weight and add bias
        if self.LC_Bias is not None: 
            x = x.multiply(self.LC_Weight).add(self.LC_Bias)
        else:
            x = x.multiply(self.LC_Weight)
        #output shape (batch_size, out_channels, sequence_length(max_block_length),n)

        #if the last dimension is 1: meaning the input is a entire sequence without blocking
        if x.size(-1) == 1: 
            ###unfold the x to the shape of (batch_size, out_channels*kernel_size,nKernels)
            x = self.locally_connection(x)
            ###convert to the shape of (batch_size, out_channels, karnel_size, nKernels)
            x = x.view(x.size(0), self.channels, self.kernel_size[0], -1)

        ###sum the value inside each kernel to the shape of (batch_size, out_channels, 1, n)
        x = x.sum(-2).unsqueeze(-2)
        ###switch the shape of (batch_size, out_channels, n, 1)
        return x.permute(0, 1, 3, 2)
    

class block_connected_layer(nn.Module):
    """
    A locally connected layer with three modes:
    1. fixed kernel size and stride and size==stride;
    2. Fixed blocks with kernel size and stride, defined by external indexes;
    3. Flexible kernel size and stride, defined by external indexes, and kernels would not overlap
    """
    def __init__(self, d_model, out_channels, kernel_size, stride, padding):
        super(block_connected_layer, self).__init__()
        self.padding = zero_padding(padding)
        self.locally_connected = locally_connected_layer(d_model, out_channels, kernel_size, use_bias=True)

    def forward(self, x):
        x = self.padding(x)
        return self.locally_connected(x)


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
        self.local_conv = locally_connected_layer(1, d_model, 3, 1, 0)
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
    def __init__(self, d_model,args, dropout=0.1,optimizer="Adam",lr=0.001):
        super(Transformer, self).__init__()
        self.args = args
        num_heads, d_ff, num_layers= args.num_heads, args.width, args.depth
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.fc = nn.Linear(d_model, 1)
        self.optimizer = optimizer
        self.init_lr = lr
    
    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.fc(x)
        return x
    
    def summary(self):
        # Print model summary by layers
        summary(self, (1000, 1))
        return
    
    def fit(self, x, y, epochs, lr=0.001):
        optimizer = {"Adam":optim.Adam, "SGD":optim.SGD, "Rmsprop":optim.RMSprop}[self.optimizer](self.parameters(), lr=lr)
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


MODELS = {
    "MLP": MLP,
    "Numeric CNN": NCNN,
    "Transformer":TransformerModel,
    "Attention CNN": AttentionCNN,
}

def main():
    print("Main function from ClassModel.py")
    #tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()
