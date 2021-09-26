import glob     #for checking dir content
import os       #for dir creation
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling1D, Flatten, Dense, Conv1D, ReLU
from keras.layers import Dropout
import matplotlib.pyplot as plt
import tensorflow as tf

GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"
TRAIN_PATH = "E:\learning resource\PhD\sugarcane\/2016_TCHBlup_2000.csv"
VALID_PATH = "E:\learning resource\PhD\sugarcane\/2017_TCHBlup_2000.csv"
LABEL_COLUMN = 'TCHBlup'

def modelling(n_layers,n_units,input_shape):

    model = Sequential()
    model.add(Conv1D(64,kernel_size=3,strides=1,padding='valid',input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64,kernel_size=3,strides=1,padding='valid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    for layers in range(n_layers):
        model.add(Dense(n_units,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation="linear"))
    model.compile(optimizer="rmsprop",loss="mean_squared_error")

    return model


def plot_loss_history(h, title):
    plt.plot(h.history['loss'], label = "Train loss")
    plt.plot(h.history['val_loss'], label = "Validation loss")
    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    plt.show()

def get_data(x,geno):
    return geno[geno["sample"]==x,]

def main():

    #genos = pd.read_csv(GENO_PATH,sep="\t").T
    #phenos = pd.read_csv(PHENO_PATH,sep="\t")
    #genos.index = genos.index.set_names("sample")
    #genos = genos.reset_index()
    #print(phenos["Clone"])
    #print(phenos.columns)
    #print(genos.shape)
    #phenos["geno"] = phenos["Clone"].apply(lambda x:genos[genos["sample"]==x])
    #phenos.head(1)

    #traits = ["CCSBlup", "TCHBlup", "FibreBlup"][1]

    train_set = tf.data.experimental.make_csv_dataset(TRAIN_PATH,64,label_name=LABEL_COLUMN)
    test_set = tf.data.experimental.make_csv_dataset(VALID_PATH,64,label_name=LABEL_COLUMN)

    examples, labels = next(iter(train_set))  # 第一个批次
    print("EXAMPLES: \n", examples, "\n")
    print("LABELS: \n", labels)




    model = modelling(3,5,train_set.shape)
    print(model.summary())


if __name__ == "__main__":
    main()