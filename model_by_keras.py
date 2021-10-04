import glob     #for checking dir content
import os       #for dir creation
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling1D, Flatten, Dense, Conv1D, ReLU
from keras.layers import Dropout
import keras.metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"

TRAIN_PATH = "E:/learning resource/PhD/sugarcane/2015_TCHBlup_2000.csv"
VALID_PATH = "E:/learning resource/PhD/sugarcane/2016_TCHBlup_2000.csv"
LABEL_COLUMN = 'TCHBlup'
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

def modelling(n_layers,n_units,input_shape):

    model = Sequential()
    model.add(Conv1D(64,kernel_size=3,strides=1,padding='valid',activation='elu',input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64,kernel_size=3,strides=1,padding='valid',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.2))
    model.add(Conv1D(32, kernel_size=3, strides=1, padding='valid',activation='elu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(16, kernel_size=3, strides=1, padding='valid',activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    for layers in range(n_layers):
        model.add(Dense(n_units,activation="elu"))
    model.add(Dropout(0.2))
    #model.add(Dense(n_layers,activation="linear"))
    #model.add(Dense(n_layers, activation="linear"))
    model.add(Dense(1, activation="linear"))
    tf.keras.optimizers.RMSprop(learning_rate=0.00001)
    model.compile(optimizer="rmsprop",loss="mean_squared_error")

    """
    Optimizers: Adam, RMSProp, Momentum 
    """

    return model


def plot_loss_history(h, title):
    plt.plot(h.history['loss'], label = "Train loss")
    plt.plot(h.history['val_loss'], label = "Validation loss")
    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    plt.show()

def main():

    #prepare data from csv files
    train_data = pd.read_csv(TRAIN_PATH,sep="\t")  #.drop(columns="Region")
    valid_data = pd.read_csv(VALID_PATH,sep="\t")  #.drop(columns="Region") The final valid data

    """
    Drop sampling index 
    """
    train_data.drop(train_data.columns[0],axis=1, inplace=True)
    valid_data.drop(valid_data.columns[0], axis=1, inplace=True)


    print("Train data:")
    print(train_data.head(3))
    print(train_data.iloc[:,0])
    print("Valid data:")
    print(valid_data.head(3))

    #pro-process data, add dim and select features

    train_targets = train_data["TCHBlup"].values   #Get the target values from train set
    train_features = train_data.iloc[:,2:]
    #ohe = OneHotEncoder()
    #ohe.fit(train_features)  # hot encode region data
    #train_features = ohe.transform(train_features)

    n_features = train_features.shape[1]
    train_features = np.expand_dims(train_features,axis=2)

    valid_targets = valid_data["TCHBlup"].values
    valid_features = valid_data.iloc[:, 2:]
    #valid_features = ohe.transform(valid_features) # hot encode region data
    valid_features = np.expand_dims(valid_features, axis=2)

    #split train data into 2 part - train and test
    features_train, features_val, target_train, target_val = train_test_split(train_features, train_targets,test_size=0.2)

    input_size = (n_features, 1)
    val_loss = 200
    while val_loss >= 70:
        model = modelling(n_layers=3, n_units=8, input_shape=input_size)
        print(model.summary())
        history = model.fit(
            features_train, target_train,
            epochs=50,
            validation_data=(features_val, target_val), verbose=1)
        plot_loss_history(history, "TCHBlup")

        # let's just print the final loss
        print(' - train loss     : ' + str(history.history['loss'][-1]))
        print(' - validation loss: ' + str(history.history['val_loss'][-1]))
        val_loss = history.history['val_loss'][-1]

        y_pred = np.reshape(model.predict(valid_features), (2000,))
        #print(y_pred.shape, valid_targets.shape)
        print("Predicted: ",y_pred)
        print("observed: ",valid_targets)
        accuracy = np.corrcoef(y_pred, valid_targets)
        print("accuracy (measured as Pearson's correlation) is: ", accuracy)
        if history.history['val_loss'][-1] < val_loss:
            json = model.to_json()
            with open("E:/learning resource/PhD/keras_models/sep_TCHBlup_model.json", "w") as file:
                file.write(json)
            model.save_weights("E:/learning resource/PhD/keras_models/sep_TCHBlup_model.json.h5")
            val_loss = history.history['val_loss'][-1]


if __name__ == "__main__":
    main()