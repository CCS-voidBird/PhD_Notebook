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

def main():

    args = get_args()
    config_path = os.path.abspath(args.config)
    print("Get config file path: ",config_path)
    config = configparser.ConfigParser()
    if platform.system().lower() == "windows":
        config.read("./MLP_parameters.ini")
    else:
        config.read(config_path)
    traits = config["BASIC"]["traits"].split("#")
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

    sil_mode = 1
    if args.silence == True:
        sil_mode = 0

    global PATH
    PATH = locat
    train_year = get_years(args.train)
    valid_year = get_years(args.valid)
    if args.sample != "all":
        sample_size = "_" + args.sample
    else:
        sample_size = "_" + args.sample

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

    print("Removing sampling index...")
    geno_data.drop(geno_data.columns[0], axis=1, inplace=True)
    print(geno_data.columns)
    geno_data = decoding(geno_data)

    filtered_data = read_pipes(geno_data, pheno_data, train_year+valid_year)

    dropout = config["BASIC"]["drop"].split("#") + ['Sample']
    keeping = [x for x in pheno_data.columns if x not in traits + config["BASIC"]["drop"].split("#")]
    keeping.remove("Series")
    print("Removing useless non-genetic factors: {}".format(dropout))
    filtered_data.drop(dropout,axis=1,inplace=True)
    print("Below factors will be fed into models: ")
    print(keeping)

    train_data = filtered_data.query('Series in @train_year').drop(["Series"],axis=1)
    valid_data = filtered_data.query('Series in @valid_year').drop(["Series"],axis=1)



    for trait in traits:

        """
        train_path = [par_path + year + "_" + trait + sample_size + ".csv" for year in train_year]
        train_data = pd.concat([pd.read_csv(path, sep="\t") for path in train_path],axis=0)  # .drop(columns="Region")
        valid_path = [par_path + year + "_" + trait + sample_size + ".csv" for year in valid_year]
        valid_data = pd.concat([pd.read_csv(path, sep="\t") for path in valid_path],axis=0)  # .drop(columns="Region") The final valid data
        """

        """
        Drop sampling index 
        """

        print("Train data:")
        print(train_data.head(3))
        print("Valid data:")
        print(valid_data.head(3))

        # pro-process data, add dim and select features

        in_train = train_data.dropna(subset=[trait], axis=0)
        in_valid = valid_data.dropna(subset=[trait], axis=0)

        train_targets = in_train[trait].values  # Get the target values from train set
        valid_targets = in_valid[trait].values
        train_features = in_train.drop(traits,axis=1)
        valid_features = in_valid.drop(traits,axis=1)


        print("currently the training method is: ",args.method)
        if "CNN" in args.method:
            print(train_features.columns)
            print("USE CNN MODEL as training method")
            if args.onehot == 1:
                print("Import One-hot encoding method.")
                train_features.replace(0.01, 3, inplace=True)
                valid_features.replace(0.01, 3, inplace=True)
                for dataset in [train_features, valid_features]:
                    for factor in keeping:
                        dataset[factor] = label_encoder.fit_transform(dataset[factor])
                if args.method == "TDCNN":
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

        #train_features[train_features == 0.01] = 3
        #valid_features[valid_features == 0.01] = 3

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
        for layers in config[args.method]["n_layers"].split(","):
            for units in config[args.method]["n_units"].split(","):
                print(layers,units)
                accs = []
                in_year_accs = []
                round = 0
                mses = []
                runtimes = []
                while round < args.round:
                    startTime = datetime.now()
                    print("Start.")
                    print(input_size)
                    model = modelling(n_layers=int(layers),
                                      n_units=int(units), input_shape=input_size,
                                      lr=float(config[args.method]["lr"]))
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
                        with open("{}{}_{}_model.json".format(model_path, trait, args.optimizer), "w") as file:
                            file.write(json)
                        model.save_weights("{}{}_{}_model.json.h5".format(model_path, trait, args.optimizer))
                    round += 1
                    accs.append(accuracy_future)
                    in_year_accs.append(accuracy)
                    runtimes.append(runtime)
                    mses.append(mse)
                    results.append([trait,args.train,args.valid,layers,units,'N/A',accuracy,accuracy_future,mse,runtime.seconds/60])
                print("The Mean accuracy of {} model is: ".format(trait), np.mean(accs))
        #["trait", "trainSet", "validSet", "n_layers", "n_units", "cnn_layers", "in_year_accuracy","predict_accuracy", "mse"]
                record_summary.append([trait,args.train,args.valid,layers,units,'N/A',np.mean(in_year_accs),np.mean(accs),np.mean(mses),np.mean(runtimes).seconds/60])

    record_train_results(results,record_columns,method=args.method,path = record_path)
    record_train_results(record_summary,record_columns,args.method,path=record_path,extra="_summary")
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
