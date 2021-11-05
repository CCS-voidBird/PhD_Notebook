import keras
import pydot
import graphviz
from keras.models import Sequential
from keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D
from keras.layers import Dropout
import tensorflow as tf
import keras.metrics

def CNN(n_layers,n_units,input_shape,optimizer="rmsprop",lr=0.00001):

    model = Sequential()
    """
    Convolutional Layers
    """
    model.add(Conv1D(64,kernel_size=5,strides=3,padding='valid',activation='elu',input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, kernel_size=3, strides=1, padding='valid',activation='elu'))
    model.add(MaxPooling1D(pool_size=2))

    # Randomly dropping 20%  sets input units to 0 each step during training time helps prevent overfitting
    model.add(Dropout(rate = 0.2))

    model.add(Flatten())

    # Full connected layers, classic multilayer perceptron (MLP)
    for layers in range(n_layers):
        model.add(Dense(n_units,activation="elu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="linear")) # The output layer uses a linear function to predict traits.
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

    """
    Optimizers: Adam, RMSProp, SGD 
    """

    return model

def MLP(n_layers,n_units,input_shape,optimizer="rmsprop",lr=0.00001):
    model = Sequential()
    model.add(Dense(n_units, activation="elu",input_shape=input_shape))
    for layers in range(n_layers):
        model.add(Dense(n_units, activation="elu"))
    #model.add(Dropout(0.2))

    model.add(Dense(1, activation="linear"))

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


def main():
    model = CNN(n_layers=3,n_units=8,input_shape=[26084,4])
    tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()