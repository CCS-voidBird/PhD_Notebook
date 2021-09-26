import glob     #for checking dir content
import os       #for dir creation
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling1D, Flatten, Dense, Conv1D, ReLU
from keras.layers import Dropout
import matplotlib.pyplot as plt

GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"

def modelling(n_layers,n_units,input_units):

    model = Sequential()
    model.add(Conv1D(64,kernel_size=3,strides=1,padding='valid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64,kernel_size=3,strides=1,padding='valid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    for layers in n_layers:
        model.add(Dense(n_units,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation="linear"))

    return model


def plot_loss_history(h, title):
    plt.plot(h.history['loss'], label = "Train loss")
    plt.plot(h.history['val_loss'], label = "Validation loss")
    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    plt.show()