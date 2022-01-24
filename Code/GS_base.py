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

    def __init__(self,silence=0):
        self.train_data = None
        self.valid_data = None
        self.method = None
        self.modelling = None
        self.silence_mode = silence
        self.sub_selection = None
        self.keeping = None
        self.traits = None
        self.records = None
        self.subset_index = None
        self.config = None
        self.save = True
        self.plot = False

    def get_data(self, configer):
        self.config = configer
        self.method = configer["BASIC"]["method"]
        self.silence_mode = configer["BASIC"]["silence"]
        self.modelling = METHODS[self.method]
        self.traits = configer["BASIC"]["traits"].split("#")
        self.sub_selection = configer["BASIC"]["sub_selection"]
        self.keeping = configer["BASIC"]["non_genetic_factors"].split("#")
        train_year = get_years(configer["BASIC"]["train"])
        valid_year = get_years(configer["BASIC"]["valid"])

        print("Getting data from config file path..")

        try:
            geno_data = pd.read_csv(self.config["PATH"]["genotype"], sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
            pheno_data = pd.read_csv(self.config["PATH"]["phenotype"], sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
        except:
            try:
                print("Using backup path (for trouble shooting)")
                print(self.config["BACKUP_PATH"]["genotype"])
                geno_data = pd.read_csv(self.config["BACKUP_PATH"]["genotype"],
                                        sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
                pheno_data = pd.read_csv(self.config["BACKUP_PATH"]["phenotype"],
                                         sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
            except:
                print("No valid path found.")
                exit()

        print(geno_data.columns)
        non_genetic_factors = [x for x in pheno_data.columns if x not in self.traits]
        print("Detected non-genetic factors from phenotype file: ",non_genetic_factors)

        geno_data = decoding(geno_data)

        filtered_data = read_pipes(geno_data, pheno_data, train_year + valid_year)

        if self.config["BASIC"]["sub_selection"] == '1':
            print("The sub_selection is enabled, thus switch to pure genetic prediction mode.")
            self.subset_index = pheno_data.Region.unique()  # Record subselection index (e.g. for region:N, B..
        else:
            print("The sub_selection is disabled..")
            print("Below factors will be fed into models: ")
            print(self.keeping) # Useful non_genetic factors e.g.   Series, Region and other..

        dropout = [x for x in non_genetic_factors if
                   x not in self.keeping and x != "Series"] + ["Sample"] # config["BASIC"]["drop"].split("#") + ['Sample']
        print("Removing useless non-genetic factors: {}".format(dropout))
        filtered_data.drop(dropout, axis=1, inplace=True)

        self.train_data = filtered_data.query('Series in @train_year').drop(["Series"], axis=1)
        self.valid_data = filtered_data.query('Series in @valid_year').drop(["Series"], axis=1)

        return

    def prepare_training(self,trait,other_factor = 'Region',factor_value = 'all'):

        if self.config["BASIC"]["sub_selection"] == '1' and factor_value != 'all':
            print("Creating subsets by non_genetic_factors..")
            print("The reference factor index: {} in {}".format(factor_value,other_factor))
            in_train = self.train_data.dropna(subset=[trait], axis=0).query('Region == @factor_value').drop(
                self.keeping,axis=1
            )
            in_valid = self.valid_data.dropna(subset=[trait], axis=0).query('Region == @factor_value').drop(
                self.keeping,axis=1
            )

        else:
            print("Use the default setting, add non-genetic factors as attributes")
            in_train = self.train_data.dropna(subset=[trait], axis=0)
            in_valid = self.valid_data.dropna(subset=[trait], axis=0)

        label_encoder = LabelEncoder()
        train_targets = in_train[trait].values  # Get the target values from train set
        valid_targets = in_valid[trait].values
        train_features = in_train.drop(self.traits, axis=1)
        valid_features = in_valid.drop(self.traits, axis=1)

        print("currently the training method is: ", self.method)
        if self.method in CNNs:
            print(train_features.columns)
            print("USE CNN MODEL as training method")
            if self.config["BASIC"]["OneHot"] == '1':
                print("Import One-hot encoding method.")
                train_features.replace(0.01, 3, inplace=True)
                valid_features.replace(0.01, 3, inplace=True)
                if self.config["BASIC"]["sub_selection"] == '0' or factor_value != 'all':
                    print("Transfer non-genetic factors: {} into features.",format(self.keeping))
                    for dataset in [train_features, valid_features]:
                        for factor in self.keeping:
                            print(factor)
                            dataset[factor] = label_encoder.fit_transform(dataset[factor])
                    if self.method == "TDCNN":
                        print("Using a 2D CNN.")
                        train_features = factor_extender(train_features, self.keeping)
                        valid_features = factor_extender(valid_features, self.keeping)


                train_features = to_categorical(train_features)
                valid_features = to_categorical(valid_features)
            else:
                print("Currently cannot solve non-genetic factors without OneHot functions.",
                      "Meanwhile, the training model will be forced to 1DCNN.")
                print(train_features.columns)
                self.method = "CNN"
                for dataset in [train_features, valid_features]:
                    if self.config["BASIC"]["sub_selection"] != '1' or factor_value == 'all':
                        dataset.drop(self.keeping, axis=1, inplace=True)


                print(train_features.columns)
                train_features = np.expand_dims(train_features, axis=2)
                valid_features = np.expand_dims(valid_features, axis=2)

        return train_features,train_targets,valid_features,valid_targets

    def trainning(self,model_path,record_path):

        record_columns = ["trait", "trainSet", "validSet", "n_layers", "n_units", "cnn_layers", "in_year_accuracy",
                          "predict_accuracy", "mse", "runtime","Region"]
        results = []
        record_summary = []

        dataset_index = ["all"]
        for trait in self.traits:
            print("training trait: ", trait)
            if self.sub_selection == '1':
                dataset_index = self.subset_index.tolist() + dataset_index
            for setting in dataset_index:
                train_features,train_targets,valid_features,valid_targets = self.prepare_training(trait,factor_value=setting)
                n_features = train_features.shape[1:]
                print("The shape of data:", train_features.shape)

                # train_features = np.expand_dims(train_features, axis=3)

                # valid_features = ohe.transform(valid_features) # hot encode region data

                # valid_features = np.expand_dims(valid_features, axis=3)

                # split train data into 2 part - train and test
                features_train, features_val, target_train, target_val = train_test_split(train_features, train_targets,
                                                                                          test_size=0.2)

                features_train_val, features_val_val, target_train_val, target_val_val = train_test_split(features_val,
                                                                                                          target_val,
                                                                                                          test_size=0.5)
                print("The input shape:", n_features)
                #print("A preview of features: ",features_train.head(1))
                input_size = n_features
                for layers in self.config[self.method]["n_layers"].split(","):
                    for units in self.config[self.method]["n_units"].split(","):
                        print(layers, units)
                        accs = []
                        in_year_accs = []
                        round = 0
                        mses = []
                        runtimes = []
                        while round < int(self.config["BASIC"]["replicate"]):
                            startTime = datetime.now()
                            print("Start.")
                            print(input_size)
                            print("Sample size: ",train_targets.shape[0]," from {} subset.".format(setting))
                            model = self.modelling(n_layers=int(layers),
                                              n_units=int(units), input_shape=input_size,
                                              lr=float(self.config[self.method]["lr"]))
                            try:
                                print(model.summary())
                            except:
                                print("It is a sklean-Random-forest model.")
                            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
                            history = model.fit(
                                features_train, target_train,
                                epochs=int(self.config["BASIC"]["Epoch"]),
                                validation_data=(features_val_val, target_val_val), verbose=int(self.silence_mode),
                                callbacks=[callback])

                            if self.plot is True:
                                plot_loss_history(history, trait)

                            # let's just print the final loss
                            print(' - train loss     : ' + str(history.history['loss'][-1]))
                            print(' - validation loss: ' + str(history.history['val_loss'][-1]))
                            print(' - loss decrease rate in last 5 epochs: ' + str(
                                np.mean(np.gradient(history.history['val_loss'][-5:]))))
                            print("Train End.")
                            endTime = datetime.now()
                            runtime = endTime - startTime
                            print("Runtime: ", runtime.seconds / 60, " min")
                            print("Predicting valid set..")
                            length = target_train_val.shape[0]
                            val_length = valid_targets.shape[0]
                            y_pred = np.reshape(model.predict(features_train_val), (length,))
                            y_pred_future = np.reshape(model.predict(valid_features), (val_length,))
                            print("Testing prediction:")
                            print("Predicted: ", y_pred_future[:10])
                            print("observed: ", target_train_val[:10])
                            accuracy = np.corrcoef(y_pred, target_train_val)[0, 1]
                            accuracy_future = np.corrcoef(y_pred_future, valid_targets)[0, 1]
                            mse = mean_squared_error(y_pred_future, valid_targets)
                            print("In-year accuracy (measured as Pearson's correlation) is: ", accuracy)
                            print("Future prediction accuracy (measured as Pearson's correlation) is: ",
                                  accuracy_future)
                            if self.save is True:
                                json = model.to_json()
                                with open("{}{}_{}_{}_model.json".format(model_path, trait, self.method,setting), "w") as file:
                                    file.write(json)
                                model.save_weights("{}{}_{}_{}_model.json.h5".format(model_path, trait, self.method,setting))
                            round += 1
                            accs.append(accuracy_future)
                            in_year_accs.append(accuracy)
                            runtimes.append(runtime)
                            mses.append(mse)
                            training_record = [trait, self.config["BASIC"]["train"], self.config["BASIC"]["valid"], layers, units, 'N/A',
                                 accuracy,
                                 accuracy_future, mse, runtime.seconds / 60, setting]
                            print(training_record)
                            results.append(training_record)
                        print("The Mean accuracy of {} model for {} sample is: ".format(trait,setting), np.mean(accs))
                        # ["trait", "trainSet", "validSet", "n_layers", "n_units", "cnn_layers", "in_year_accuracy","predict_accuracy", "mse"]
                        record_summary.append(
                            [trait, self.config["BASIC"]["train"], self.config["BASIC"]["valid"], layers, units, 'N/A',
                             np.mean(in_year_accs), np.mean(accs), np.mean(mses), np.mean(runtimes).seconds / 60, setting])

            record_train_results(results, record_columns, method=self.method, path=record_path)
            record_train_results(record_summary, record_columns, self.method, path=record_path, extra="_summary")
            check_usage()





def main():

    args = get_args()
    config_path = os.path.abspath(args.config)
    print("Get config file path from: ",config_path)
    config = configparser.ConfigParser()
    if platform.system().lower() == "windows":
        print(config_path)
        config.read(config_path)
        #print(config["BACKUP_PATH"]["genotype"])
    else:
        config.read(config_path)

    """
    Create folders from given output path
    """
    if config["OUTPUT"]["output"][0] == "/":
        locat = '/' + config["OUTPUT"]["output"].strip('/') + '/'
    else:
        locat = config["OUTPUT"]["output"].strip('/') + '/'
    if not os.path.exists(locat):
        os.mkdir(locat)
    locat = locat + "{}_vs_{}/".format(config["BASIC"]["train"], config["BASIC"]["valid"])
    model_path = locat + "models/"
    record_path = locat + "records_{}/".format(args.optimizer)
    for path in [locat,model_path,record_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    composer = ML_composer()
    composer.get_data(config)
    composer.trainning(model_path=model_path,record_path=record_path)

if __name__ == "__main__":
    main()
