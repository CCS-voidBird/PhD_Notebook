import glob     #for checking dir content
import os       #for dir creation
from Functions import *
from GSModel import *
import pandas as pd
import configparser
import keras
from keras.models import Sequential
from keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D
from keras.layers import Dropout
import keras.metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

##############################################################
##########Training requirement################################
##Training by para-sets -- convolutional act function + full connected act function + optimizer + learningRate###
##Output format: an table with mean accuracy for each para set; A density plot for each accuracy##########
#################################################################

"""
GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"

TRAIN_PATH = "E:/learning resource/PhD/sugarcane/2015_TCHBlup_2000.csv"
VALID_PATH = "E:/learning resource/PhD/sugarcane/2016_TCHBlup_2000.csv"
"""

METHODS = {
    "MLP": MLP,
    "CNN": CNN,
    "TDCNN": TDCNN
}


#sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0 )
"""
def modelling(n_layers,n_units,input_shape,optimizer="rmsprop",lr=0.00001):

    model = Sequential()
    model.add(Conv1D(64,kernel_size=5,strides=3,padding='valid',activation='elu',input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, kernel_size=3, strides=1, padding='valid',activation='elu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate = 0.2))

    model.add(Flatten())
    for layers in range(n_layers):
        model.add(Dense(n_units,activation="elu"))
    model.add(Dropout(0.2))
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

    
    #Optimizers: Adam, RMSProp, SGD 
    

    return model
"""

def plot_loss_history(h, title):
    plt.plot(h.history['loss'], label = "Train loss")
    plt.plot(h.history['val_loss'], label = "Validation loss")
    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    #plt.show()

def main():

    traits = ["TCHBlup",
             "CCSBlup",
             "FibreBlup"]

    """
    add a parameter function: T/V year, trait/all. 
    output format: a table with avg accuracy for each parameter/trait
    """
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument('-p', '--path', type=str, help="Input path.", required=True)
    req_grp.add_argument('-1', '--train', type=str, help="Input train year.", required=True)
    req_grp.add_argument('-2', '--valid', type=str, help="Input valid year.", required=True)
    req_grp.add_argument('-m', '--method', type=str, help="Select training method (CNN/MLP).", default="CNN")
    req_grp.add_argument('-o', '--output', type=str, help="Input output dir.", required=True)
    req_grp.add_argument('-s', '--sample', type=str, help="number of sample", default="all")
    req_grp.add_argument('-a', '--region', type=bool, help="add regions (T/F)", default=False)
    req_grp.add_argument('-r', '--round', type=int, help="training round.", default=20)
    req_grp.add_argument('-epo', '--epoch', type=int, help="training epoch.", default=50)
    req_grp.add_argument('-opt', '--optimizer', type=str, help='select optimizer: Adam, SGD, rmsprop',
                         default="rmsprop")
    req_grp.add_argument('-plot', '--plot', type=bool, help="give plot?",
                         default=False)
    req_grp.add_argument('-sli', '--silence', type=bool, help="silent mode",
                         default=True)
    req_grp.add_argument('-loss', '--loss', type=int, help="The target loss",
                         default=10)
    args = parser.parse_args()

    config = configparser.ConfigParser()
    #config.read("/clusterdata/uqcche32/MLP_parameters.ini")
    config.read("./MLP_parameters.ini")

    par_path = args.path
    modelling = METHODS[args.method]
    if args.output[0] == "/":
        locat = '/' + args.output.strip('/') + '/'
    else:
        locat = args.output.strip('/') + '/'
    if not os.path.exists(locat):
        os.mkdir(locat)
    locat = locat + "{}_vs_{}/".format(args.train, args.valid)
    model_path = locat + "models/"
    record_path = locat + "records_{}/".format(args.optimizer)
    for path in [locat,model_path,record_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    sil_mode = 0
    if args.silence == True:
        sil_mode = 1

    global PATH
    PATH = locat
    train_year = args.train.split("-")
    valid_year = args.valid.split("-")
    if args.sample != "all":
        sample_size = "_" + args.sample
    else:
        sample_size = "_" + args.sample

    in_year_record = open(record_path+"in_year_train_record_{}_vs_{}.csv".format("_".join(train_year),"_".join(valid_year)),"w")
    record = open(record_path+"train_record_{}_vs_{}.csv".format("_".join(train_year),"_".join(valid_year)),"w")
    raw_record = open(record_path+"train_record_{}_vs_{}_raw.csv".format("_".join(train_year),"_".join(valid_year)),"w")
    accs = {"TCHBlup": [], "CCSBlup": [], "FibreBlup": []}
    in_year_accs = {"TCHBlup": [], "CCSBlup": [], "FibreBlup": []}
    label_encoder = LabelEncoder()
    for trait in traits:
        # prepare data from csv files
        train_path = [par_path + year + "_" + trait + sample_size + ".csv" for year in train_year]
        train_data = pd.concat([pd.read_csv(path, sep="\t") for path in train_path],axis=0)  # .drop(columns="Region")
        valid_path = [par_path + year + "_" + trait + sample_size + ".csv" for year in valid_year]
        valid_data = pd.concat([pd.read_csv(path, sep="\t") for path in valid_path],axis=0)  # .drop(columns="Region") The final valid data

        """
        Drop sampling index 
        """

        train_data.drop(train_data.columns[0], axis=1, inplace=True)
        valid_data.drop(valid_data.columns[0], axis=1, inplace=True)

        print("Train data:")
        print(train_data.head(3))
        print("Valid data:")
        print(valid_data.head(3))

        # pro-process data, add dim and select features


        train_targets = train_data[trait].values  # Get the target values from train set
        valid_targets = valid_data[trait].values
        train_features = train_data.iloc[:, 2:]
        valid_features = valid_data.iloc[:, 2:]
        print("currently the training method is: ",args.method)
        if "CNN" in args.method:
            print(train_features.columns)
            factors = []
            print("USE CNN MODEL as training method")
            train_features.replace(0.01, 3, inplace=True)
            valid_features.replace(0.01, 3, inplace=True)

            if args.region is True:
                factors.append("Region")
                train_features["Region"] = train_data["Region"]
                valid_features["Region"] = valid_data["Region"]
                for dataset in [train_features, valid_features]:
                    dataset["Region"] = label_encoder.fit_transform(dataset["Region"])
            train_features = factor_extender(train_features,factors)
            valid_features = factor_extender(valid_features,factors)

            train_features = to_categorical(train_features)
            valid_features = to_categorical(valid_features)

        #train_features[train_features == 0.01] = 3
        #valid_features[valid_features == 0.01] = 3



        n_features = train_features.shape[1:]
        print(train_features.shape)

        #train_features = np.expand_dims(train_features, axis=3)

        # valid_features = ohe.transform(valid_features) # hot encode region data

        #valid_features = np.expand_dims(valid_features, axis=3)

        # split train data into 2 part - train and test
        features_train, features_val, target_train, target_val = train_test_split(train_features, train_targets,
                                                                                  test_size=0.2)

        features_train_val, features_val_val, target_train_val, target_val_val = train_test_split(features_val,
                                                                                                  target_val,
                                                                                                  test_size=0.5)
        print(n_features)
        input_size = n_features
        round = 0

        while round < args.round:
            print(input_size)
            model = modelling(n_layers=int(config[args.method]["n_layers"]), n_units=int(config[args.method]["n_units"]), input_shape=input_size,lr=config[args.method]["lr"])
            try:
                print(model.summary())
            except:
                print("It is a sklean-Random-forest model.")
            history = model.fit(
                features_train, target_train,
                epochs=args.epoch,
                validation_data=(features_val_val, target_val_val), verbose=sil_mode)
            if args.plot is True:
                plot_loss_history(history, trait)

            # let's just print the final loss
            print(' - train loss     : ' + str(history.history['loss'][-1]))
            print(' - validation loss: ' + str(history.history['val_loss'][-1]))
            length = target_train_val.shape[0]
            val_length = valid_targets.shape[0]
            y_pred = np.reshape(model.predict(features_train_val), (length,))
            y_pred_future = np.reshape(model.predict(valid_features), (val_length,))

            print("Predicted: ", y_pred[:10])
            print("observed: ", target_train_val[:10])
            accuracy = np.corrcoef(y_pred, target_train_val)[0, 1]
            accuracy_future = np.corrcoef(y_pred_future, valid_targets)[0, 1]
            print("In-year accuracy (measured as Pearson's correlation) is: ", accuracy)
            print("Future prediction accuracy (measured as Pearson's correlation) is: ", accuracy_future)
            if history.history['val_loss'][-1] < args.loss:
                json = model.to_json()
                with open("{}{}_{}_model.json".format(model_path,trait,args.optimizer), "w") as file:
                    file.write(json)
                model.save_weights("{}{}_{}_model.json.h5".format(model_path,trait,args.optimizer))
            round += 1
            accs[trait].append(accuracy_future)
            in_year_accs[trait].append(accuracy)

        print("The Mean accuracy of {} model is: ".format(trait), np.mean(accs[trait]))
    for key in accs.keys():
        record.write("{}\t{}\n".format(key,np.mean(accs[key])))
        raw_record.write("{}\t{}\n".format(key,"\t".join([str(x) for x in accs[key]])))
        in_year_record.write("{}\t{}\n".format(key,"\t".join([str(x) for x in in_year_accs[key]])))
    record.close()
    raw_record.close()
    in_year_record.close()

if __name__ == "__main__":
    main()