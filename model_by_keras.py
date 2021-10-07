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
import argparse
from sklearn.preprocessing import OneHotEncoder

GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"

TRAIN_PATH = "E:/learning resource/PhD/sugarcane/2015_TCHBlup_2000.csv"
VALID_PATH = "E:/learning resource/PhD/sugarcane/2016_TCHBlup_2000.csv"
LABEL_COLUMN = 'TCHBlup'
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

def modelling(n_layers,n_units,input_shape):

    model = Sequential()
    model.add(Conv1D(64,kernel_size=5,strides=1,padding='valid',activation='elu',input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(64,kernel_size=3,strides=1,padding='valid',activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(8, kernel_size=3, strides=1, padding='valid',activation='elu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
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
    #plt.show()

def main():

    train_year = "2015"
    valid_year = "2016"

    folder = "E:/learning resource/PhD/sugarcane/"

    traits = ["TCHBlup",
             "CCSBlup",
             "FibreBlup"]

    """
    add a parameter function: T/V year, trait/all. 
    output format: a table with avg accuracy for each parameter/trait
    """
    # python model_by_keras.py -p "E:/learning resource/PhD/sugarcane/" -1 2016 -2 2016 -o ../new_model/test/ -s 400
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument('-p', '--path', type=str, help="Input path.", required=True)
    req_grp.add_argument('-1', '--train', type=str, help="Input train year.", required=True)
    req_grp.add_argument('-2', '--valid', type=str, help="Input valid year.", required=True)
    req_grp.add_argument('-o', '--output', type=str, help="Input output dir.", required=True)
    req_grp.add_argument('-s', '--sample', type=str, help="number of sample", default="all")
    req_grp.add_argument('-a', '--region', type=bool, help="add regions (T/F)", default=False)
    req_grp.add_argument('-r', '--round', type=int, help="training round.", default=20)
    args = parser.parse_args()
    par_path = args.path
    if args.output[0] == "/":
        locat = '/' + args.output.strip('/') + '/'
    else:
        locat = args.output.strip('/') + '/'
    os.system("mkdir -p {}".format(locat))
    global PATH
    PATH = locat
    train_year = args.train
    valid_year = args.valid
    if args.sample != "all":
        sample_size = "_" + args.sample
    else:
        sample_size = ""

    record = open(locat+"train_record_{}_vs_{}.csv".format(train_year,valid_year),"a")
    accs = {"TCHBlup": [], "CCSBlup": [], "FibreBlup": []}

    for trait in traits:
        train_path = par_path + train_year + "_" + trait + sample_size + ".csv"
        valid_path = par_path + valid_year + "_" + trait + sample_size + ".csv"
        # prepare data from csv files
        train_data = pd.read_csv(train_path, sep="\t")  # .drop(columns="Region")
        valid_data = pd.read_csv(valid_path, sep="\t")  # .drop(columns="Region") The final valid data

        """
        Drop sampling index 
        """

        train_data.drop(train_data.columns[0], axis=1, inplace=True)
        valid_data.drop(valid_data.columns[0], axis=1, inplace=True)

        print("Train data:")
        print(train_data.head(3))
        print(train_data.iloc[:, 0])
        print("Valid data:")
        print(valid_data.head(3))

        # pro-process data, add dim and select features

        train_targets = train_data[trait].values  # Get the target values from train set
        train_features = train_data.iloc[:, 2:]
        # ohe = OneHotEncoder()
        # ohe.fit(train_features)  # hot encode region data
        # train_features = ohe.transform(train_features)

        n_features = train_features.shape[1]
        train_features = np.expand_dims(train_features, axis=2)

        valid_targets = valid_data[trait].values
        valid_features = valid_data.iloc[:, 2:]
        # valid_features = ohe.transform(valid_features) # hot encode region data
        valid_features = np.expand_dims(valid_features, axis=2)

        # split train data into 2 part - train and test
        features_train, features_val, target_train, target_val = train_test_split(train_features, train_targets,
                                                                                  test_size=0.2)

        features_train_val, features_val_val, target_train_val, target_val_val = train_test_split(features_val,
                                                                                                  target_val,
                                                                                                  test_size=0.5)

        input_size = (n_features, 1)
        round = 0

        while round < args.round:
            model = modelling(n_layers=3, n_units=8, input_shape=input_size)
            print(model.summary())
            history = model.fit(
                features_train, target_train,
                epochs=50,
                validation_data=(features_val_val, target_val_val), verbose=1)
            plot_loss_history(history, "TCHBlup")

            # let's just print the final loss
            print(' - train loss     : ' + str(history.history['loss'][-1]))
            print(' - validation loss: ' + str(history.history['val_loss'][-1]))
            val_loss = history.history['val_loss'][-1]
            length = target_train_val.shape[0]
            y_pred = np.reshape(model.predict(features_train_val), (length,))
            y_pred_future = np.reshape(model.predict(valid_features), (2000,))
            # print(y_pred.shape, valid_targets.shape)
            print("Predicted: ", y_pred[:10])
            print("observed: ", target_train_val[:10])
            accuracy = np.corrcoef(y_pred, target_train_val)[0, 1]
            accuracy_future = np.corrcoef(y_pred_future, valid_targets)[0, 1]
            print("In-year accuracy (measured as Pearson's correlation) is: ", accuracy)
            print("In-year accuracy (measured as Pearson's correlation) is: ", accuracy_future)
            if history.history['val_loss'][-1] < val_loss:
                json = model.to_json()
                with open("E:/learning resource/PhD/keras_models/sep_TCHBlup_model.json", "w") as file:
                    file.write(json)
                model.save_weights("E:/learning resource/PhD/keras_models/sep_TCHBlup_model.json.h5")
            round += 1
            accs[trait].append(accuracy_future)

        print("The Mean accuracy of {} model is: ".format(trait), np.mean(accs[trait]))
    for key in accs.keys():
        record.write("{}\t{}\n".format(key,np.mean(accs[key])))

    record.close()


if __name__ == "__main__":
    main()