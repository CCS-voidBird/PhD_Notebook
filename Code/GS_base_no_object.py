import glob     #for checking dir content
import os       #for dir creation
from Functions import *
from GSModel import *
import pandas as pd
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
import platform
from datetime import datetime
from sklearn.metrics import mean_squared_error
import configparser

##############################################################
##########Training requirement################################
##Training by para-sets -- convolutional act function + full connected act function + optimizer + learningRate###
##Output format: an table with mean accuracy for each para set; A density plot for each accuracy##########
#################################################################

CNNs = ["CNN","TDCNN","DeepGS"]

"""
GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"

TRAIN_PATH = "E:/learning resource/PhD/sugarcane/2015_TCHBlup_2000.csv"
VALID_PATH = "E:/learning resource/PhD/sugarcane/2016_TCHBlup_2000.csv"
"""

def plot_loss_history(h, title):
    plt.plot(h.history['loss'], label = "Train loss")
    plt.plot(h.history['val_loss'], label = "Validation loss")
    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    #plt.show()

class ML_composer:

    def __init__(self,train_data,valid_data,selected_model):
        self.train_data = train_data
        self.valid_data = valid_data
        self.model = selected_model
        

def main():

    args = get_args()
    config_path = os.path.abspath(args.config)
    print("Get config file path from: ",config_path)
    config = configparser.ConfigParser()
    if platform.system().lower() == "windows":
        config.read("./MLP_parameters.ini")
    else:
        config.read(config_path)
    traits = config["BASIC"]["traits"].split("#")
    selected_model = config["BASIC"]["method"]
    modelling = METHODS[selected_model]

    """
    Create folders from given output path
    """
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

    """
    select silence mode
    """

    sil_mode = 1
    if args.silence == True:
        sil_mode = 0

    global PATH
    PATH = locat

    """
    backup year-select codes
    train_year = get_years(args.train)
    valid_year = get_years(args.valid)
    """

    train_year = get_years(config["BASIC"]["train"])
    valid_year = get_years(config["BASIC"]["valid"])

    #["trait","trainSet","validSet","n_features","test_score","valid_score","accuracy","mse"]
    record_columns = ["trait","trainSet","validSet","n_layers","n_units","cnn_layers","in_year_accuracy",
                      "predict_accuracy","mse","runtime"]
    results = []
    record_summary = []
    label_encoder = LabelEncoder()

    geno_data = None
    pheno_data = None

    try:
        geno_data = pd.read_csv(config["PATH"]["genotype"], sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
        pheno_data = pd.read_csv(config["PATH"]["phenotype"], sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
    except:
        try:
            print("Using backup path (for trouble shooting)")
            geno_data = pd.read_csv(config["BACKUP_PATH"]["genotype"],
                                    sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
            pheno_data = pd.read_csv(config["BACKUP_PATH"]["phenotype"],
                                     sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
        except:
            print("No valid path found.")
            exit()
    """
    Have manually removed sampling index.
    print("Removing sampling index...")
    geno_data.drop(geno_data.columns[0], axis=1, inplace=True)
    """
    print(geno_data.columns)
    non_genetic_factors = [x for x in pheno_data.columns if x not in traits]

    geno_data = decoding(geno_data)

    filtered_data = read_pipes(geno_data, pheno_data, train_year+valid_year)




    keeping = config["BASIC"]["non_genetic_factors "].split("#")

    if config["BASIC"]["sub_selection"] == 1:
        print("The sub_selection is enabled, thus switch to pure genetic prediction mode.")
    else:
        print("The sub_selection is disabled..")
        print("Below factors will be fed into models: ")
        print(keeping)

    dropout = [x for x in non_genetic_factors if x not in keeping and x is not "Series"]  # config["BASIC"]["drop"].split("#") + ['Sample']
    print("Removing useless non-genetic factors: {}".format(dropout))
    filtered_data.drop(dropout,axis=1,inplace=True)

    train_data = filtered_data.query('Series in @train_year').drop(["Series"],axis=1)
    valid_data = filtered_data.query('Series in @valid_year').drop(["Series"],axis=1)



    for trait in traits:

        """
        Drop sampling index 
        """
        print("training trait: ",trait)
        print("Train data:")
        print(train_data.head(3))
        print("Valid data:")
        print(valid_data.head(3))

        # pro-process data, add dim and select features

        in_train = train_data.dropna(subset=[trait], axis=0)
        in_valid = valid_data.dropna(subset=[trait], axis=0)

        if config["BASIC"]["sub_selection"] == 1:
            print("Creating subsets by non_genetic_factors..")

            print("Removing non_genetic factors..")

        train_targets = in_train[trait].values  # Get the target values from train set
        valid_targets = in_valid[trait].values
        train_features = in_train.drop(traits,axis=1)
        valid_features = in_valid.drop(traits,axis=1)


        print("currently the training method is: ",selected_model)
        if selected_model in CNNs:
            print(train_features.columns)
            print("USE CNN MODEL as training method")
            if config["BASIC"]["OneHot"] == 1:
                print("Import One-hot encoding method.")
                train_features.replace(0.01, 3, inplace=True)
                valid_features.replace(0.01, 3, inplace=True)
                for dataset in [train_features, valid_features]:
                    for factor in keeping:
                        dataset[factor] = label_encoder.fit_transform(dataset[factor])
                if selected_model == "TDCNN":
                    print("Using a 2D CNN.")
                    train_features = factor_extender(train_features, keeping)
                    valid_features = factor_extender(valid_features, keeping)

                train_features = to_categorical(train_features)
                valid_features = to_categorical(valid_features)
            else:
                print("Currently cannot solve non-genetic factors without OneHot functions.",
                      "Meanwhile, the training model will be forced to 1DCNN.")
                for dataset in [train_features, valid_features]:
                    dataset.drop(keeping,axis=1,inplace=True)

                print(train_features.columns)
                train_features = np.expand_dims(train_features,axis=2)
                valid_features = np.expand_dims(valid_features,axis=2)


        n_features = train_features.shape[1:]
        print("The shape of data:",train_features.shape)

        #train_features = np.expand_dims(train_features, axis=3)

        # valid_features = ohe.transform(valid_features) # hot encode region data

        #valid_features = np.expand_dims(valid_features, axis=3)

        # split train data into 2 part - train and test
        features_train, features_val, target_train, target_val = train_test_split(train_features, train_targets,
                                                                                  test_size=0.2)

        features_train_val, features_val_val, target_train_val, target_val_val = train_test_split(features_val,
                                                                                                  target_val,
                                                                                                  test_size=0.5)
        print("The input shape:",n_features)
        input_size = n_features
        for layers in config[selected_model]["n_layers"].split(","):
            for units in config[selected_model]["n_units"].split(","):
                print(layers,units)
                accs = []
                in_year_accs = []
                round = 0
                mses = []
                runtimes = []
                while round < int(config["BASIC"]["replicate"]):
                    startTime = datetime.now()
                    print("Start.")
                    print(input_size)
                    model = modelling(n_layers=int(layers),
                                      n_units=int(units), input_shape=input_size,
                                      lr=float(config[selected_model]["lr"]))
                    try:
                        print(model.summary())
                    except:
                        print("It is a sklean-Random-forest model.")
                    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
                    history = model.fit(
                        features_train, target_train,
                        epochs=config["BASIC"]["Epoch"],
                        validation_data=(features_val_val, target_val_val), verbose=sil_mode,callbacks=[callback])
                    if args.plot is True:
                        plot_loss_history(history, trait)

                    # let's just print the final loss
                    print(' - train loss     : ' + str(history.history['loss'][-1]))
                    print(' - validation loss: ' + str(history.history['val_loss'][-1]))
                    print(' - loss decrease rate in last 5 epochs: ' + str(np.mean(np.gradient(history.history['val_loss'][-5:]))))
                    print("Train End.")
                    endTime = datetime.now()
                    runtime = endTime - startTime
                    print("Runtime: ",runtime.seconds/60," min")
                    print("Predicting valid set..")
                    length = target_train_val.shape[0]
                    val_length = valid_targets.shape[0]
                    y_pred = np.reshape(model.predict(features_train_val), (length,))
                    y_pred_future = np.reshape(model.predict(valid_features), (val_length,))

                    print("Predicted: ", y_pred[:10])
                    print("observed: ", target_train_val[:10])
                    accuracy = np.corrcoef(y_pred, target_train_val)[0, 1]
                    accuracy_future = np.corrcoef(y_pred_future, valid_targets)[0, 1]
                    mse = mean_squared_error(y_pred_future,valid_targets)
                    print("In-year accuracy (measured as Pearson's correlation) is: ", accuracy)
                    print("Future prediction accuracy (measured as Pearson's correlation) is: ", accuracy_future)
                    if args.save is True:
                        json = model.to_json()
                        with open("{}{}_{}_model.json".format(model_path, trait, selected_model), "w") as file:
                            file.write(json)
                        model.save_weights("{}{}_{}_model.json.h5".format(model_path, trait, selected_model))
                    round += 1
                    accs.append(accuracy_future)
                    in_year_accs.append(accuracy)
                    runtimes.append(runtime)
                    mses.append(mse)
                    results.append([trait,config["BASIC"]["train"],config["BASIC"]["valid"],layers,units,'N/A',accuracy,accuracy_future,mse,runtime.seconds/60])
                print("The Mean accuracy of {} model is: ".format(trait), np.mean(accs))
        #["trait", "trainSet", "validSet", "n_layers", "n_units", "cnn_layers", "in_year_accuracy","predict_accuracy", "mse"]
                record_summary.append([trait,config["BASIC"]["train"],config["BASIC"]["valid"],layers,units,'N/A',np.mean(in_year_accs),np.mean(accs),np.mean(mses),np.mean(runtimes).seconds/60])

    record_train_results(results,record_columns,method=selected_model,path = record_path)
    record_train_results(record_summary,record_columns,selected_model,path=record_path,extra="_summary")
    """
    in_year_record = open(record_path+"in_year_train_record_{}_vs_{}.csv".format("_".join(train_year),"_".join(valid_year)),"w")
    record = open(record_path+"train_record_{}_vs_{}.csv".format("_".join(train_year),"_".join(valid_year)),"w")
    raw_record = open(record_path+"train_record_{}_vs_{}_raw.csv".format("_".join(train_year),"_".join(valid_year)),"w")
    for key in accs.keys():
        record.write("{}\t{}\n".format(key,np.mean(accs[key])))
        raw_record.write("{}\t{}\n".format(key,"\t".join([str(x) for x in accs[key]])))
        in_year_record.write("{}\t{}\n".format(key,"\t".join([str(x) for x in in_year_accs[key]])))
    record.close()
    raw_record.close()
    in_year_record.close()
    """

if __name__ == "__main__":
    main()
