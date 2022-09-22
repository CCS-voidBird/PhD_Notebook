
from Functions import *
from GSModel import *
import pandas as pd
import matplotlib.pyplot as plt
try:
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
except:
    print("This a CPU-only platform.")
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
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
#Test command (Local): python GS_base.py --config ./test_config
CNNs = ["CNN","TDCNN","DeepGS"]
PATIENCE = 100
"""
GENO_PATH = "E:\learning resource\PhD\geno_data1.csv"
PHENO_PATH = "E:\learning resource\PhD\phenotypes.csv"

TRAIN_PATH = "E:/learning resource/PhD/sugarcane/2015_TCHBlup_2000.csv"
VALID_PATH = "E:/learning resource/PhD/sugarcane/2016_TCHBlup_2000.csv"
"""
def get_args():
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument('-g', '--ped', type=str, help="PED-like file name", required=True)
    req_grp.add_argument('-pheno', '--pheno', type=str, help="Phenotype file.", required=True)
    req_grp.add_argument('-index', '--index', type=str, help="File of train/validate reference", required=True)
    req_grp.add_argument('-m', '--model', type=str, help="Select training model.", default="CNN")
    req_grp.add_argument('-o', '--output', type=str, help="Input output dir.")
    req_grp.add_argument('-r', '--round', type=int, help="training round.", default=20)
    req_grp.add_argument('-epo', '--epoch', type=int, help="training epoch.", default=50)
    req_grp.add_argument('-plot', '--plot', type=bool, help="show plot?",
                         default=False)
    req_grp.add_argument('-sli', '--silence', type=bool, help="silent mode",
                         default=True)
    req_grp.add_argument('-save', '--save', type=bool, help="save model True/False",
                         default=False)
    req_grp.add_argument('-config', '--config', type=str, help='config file path, default: ~/MLP_parameters.ini',
                         default="~/MLP_parameters.ini",required=True)
    args = parser.parse_args()

    return args


def plot_loss_history(h, title):
    plt.plot(h.history['loss'], label = "Train loss")
    plt.plot(h.history['val_loss'], label = "Validation loss")
    plt.xlabel('Epochs')
    plt.title(title)
    plt.legend()
    #plt.show()

class ML_composer:

    def __init__(self,silence=0):
        self._raw_data = {"GENO":pd.DataFrame(),"PHENO":pd.DataFrame(),"INDEX":pd.DataFrame()}
        self.train_data = None
        self.valid_data = None
        self._model = {"INIT_MODEL":None,"TRAINED_MODEL":None}
        self.method = None
        self.modelling = None
        self.silence_mode = silence
        self.config = None
        self.save = True
        self.plot = False
        self.args = None

    def get_data(self,configer,args):
        self.args = args
        self.config = configer
        self.method = configer["BASIC"]["method"]
        self.silence_mode = configer["BASIC"]["silence"]
        self.modelling = METHODS[self.method]
        self.traits = configer["BASIC"]["traits"].split("#")
        self.keeping = configer["BASIC"]["non_genetic_factors"].split("#")
        if "None" in self.keeping: self.keeping = []
        #train_year = get_years(configer["BASIC"]["train"])
        #valid_year = get_years(configer["BASIC"]["valid"])

        print("Getting data from config file path..")

        self._raw_data["GENO"] = pd.read_table(args.ped,sep="\t",header=None)
        self._raw_data["PHENO"] = pd.read_table(args.pheno, sep="\t", header=None)
        self._raw_data["INDEX"] = pd.read_table(args.index,sep="\t", header=None)
        train_sample = self._raw_data["INDEX"].query(
            "index == 1"
        ).id.unique()

        valid_sample = self._raw_data["INDEX"].query(
            "index is not 1"
        ).id.unique()

        print("Train clones: {}, valid clones: {}".format(len(train_sample),len(valid_sample)))

        self._raw_data["INFO"] = self._raw_data["GENO"].iloc[:,0:6]


        remove_list = []
        print("Strict training model detected, detecting overlapping in training data..")
        remove_list = np.intersect1d(train_sample,valid_sample)
        #train_sample.query("id not in @remove_list")
        #valid_sample.query("id not in @remove_list")
        print("Finished")

        print("Get genotype shape:",self._raw_data["GENO"].iloc[:,6:].shape)
        print(self._raw_data["GENO"].iloc[:,6:].iloc[1:10,1:10])


        self.train_data = self._raw_data["GENO"].query('X.1 in @train_sample').query('X.1 not in @remove_list').iloc[:,6:]
        self.valid_data = self._raw_data["GENO"].query('X.1 in @valid_sample').iloc[:,6:]
        print(self.train_data.shape)
        print(self.valid_data.shape)

        #self.train_data = filtered_data.query('Series in @train_year').query('Sample not in @remove_list').drop(["Series","Sample"], axis=1)
        #self.valid_data = filtered_data.query('Series in @valid_year').drop(["Series","Sample"], axis=1)


        return

    def prepare_training(self,trait,other_factor = 'Region',factor_value = 'all',test=True):

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
            #print(train_features.columns)
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
                      "Meanwhile, the non-genetic factors will be excluded.")
                print(train_features.columns[1:10])
                print(train_features.shape)
                self.method = "CNN"
                for dataset in [train_features, valid_features]:
                    if self.config["BASIC"]["sub_selection"] != '1' or factor_value == 'all':
                        try:
                            dataset.drop(self.keeping, axis=1, inplace=True)
                        except:
                            print("The {} were already be removed".format(self.keeping))


                print(train_features.columns[1:10])
                print(train_features.shape)
                train_features = np.expand_dims(train_features, axis=2)
                valid_features = np.expand_dims(valid_features, axis=2)

        elif self.method == "MLP":
            print(train_features.columns[1:10])
            print("USE MLP MODEL as training method")
            if self.config["BASIC"]["OneHot"] == '1':
                print("Import One-hot encoding method.")
                train_features.replace(0.01, None, inplace=True)
                valid_features.replace(0.01, None, inplace=True)
                if self.config["BASIC"]["sub_selection"] == '0' or factor_value == 'all':
                    print("Transfer non-genetic factors: {} into features.", format(self.keeping))
                    for dataset in [train_features, valid_features]:
                        for factor in self.keeping:
                            print(factor)
                            dataset[factor] = label_encoder.fit_transform(dataset[factor])
                train_features = to_categorical(train_features)
                valid_features = to_categorical(valid_features)
            else:
                print("Currently cannot solve non-genetic factors without OneHot functions.",
                      "Meanwhile, the non-genetic factors will be excluded from dataset.")
                for dataset in [train_features, valid_features]:
                    if self.config["BASIC"]["sub_selection"] != '1' or factor_value == 'all':
                        dataset.drop(self.keeping, axis=1, inplace=True)
                print(train_features.columns)
                #train_features = np.expand_dims(train_features, axis=2)
                #valid_features = np.expand_dims(valid_features, axis=2)

        elif self.method == "RF":
            print("Currently only support RF training with numeric values, thus, the data will exclude non-genetic factos.")
            try:
                print("Removing non-genetic factors: ",self.keeping)
                train_features.drop(self.keeping,axis=1,inplace=True)
                valid_features.drop(self.keeping,axis=1,inplace=True)
            except:
                print("These non_genetic factors are already removed: ",self.keeping)
                print(train_features.columns[1:10])

            print("The selected region is: ",factor_value)

        train_features = np.asarray(train_features).astype(np.float32)
        train_targets = np.asarray(train_targets).astype(np.float32)
        valid_features = np.asarray(valid_features).astype(np.float32)
        valid_targets = np.asarray(valid_targets).astype(np.float32)

        return train_features,train_targets,valid_features,valid_targets

    def make_hp_set(self):
        hps = [hp for hp in self.config[self.method]]
        hp_sets = list(combi([v.split(",") for k,v in dict(self.config[self.method]).items()]))
        print("The hyper-parameter is: \n","\t".join(hps))
        for content in hp_sets:
            print("\t".join(content))
        return hps,hp_sets

    def make_forest(self,model_path,record_path):
        hps, hp_sets = self.make_hp_set()
        record_cols = ["trait", "trainSet", "validSet"]+hps+["test_score", "valid_score", "accuracy", "mse","Region", "Runtime"]
        accs = []  # pd.DataFrame(columns=["trait","trainSet","validSet","score","cov"])
        records = []
        paraList = []
        
        #max_feature_list = [int(x) for x in self.config["RM"]["max_features"].split(",")]
        """
        A tuple of RF model hp: 1st: max_features, 2nd: n_trees, 3..4.. 
        """
        for trait in self.traits:
            for region in self.subset_index+["all"]:
                print(trait)
                avg_acc = []
                avg_score = []
                acg_same_score = []
                avg_mse = []
                avg_runtime = []
                best_model = [0,None]
                for hp_set in hp_sets:
                    print("Modified hyper-parameter names: \n", "\t".join(hps))
                    print("Now training by the following hyper_parameters: ".format(hp_set))
                    r = 0
                    while r < int(self.config["BASIC"]["replicate"]):
                        startTime = datetime.now()

                        train_features, train_targets, valid_features, valid_targets = self.prepare_training(trait,
                                                                                                             factor_value=region)

                        features_train, features_val, target_train, target_val = train_test_split(train_features,
                                                                                                  train_targets,
                                                                                                  test_size=0.2)

                        features_train_val, features_val_val, target_train_val, target_val_val = train_test_split(
                            features_val,
                            target_val,
                            test_size=0.5)
                        print(features_train)

                        model = RF(specific=True,n_estimators=int(hp_set[0]), n_features=int(hp_set[1]))

                        model.fit(features_train, target_train)
                        endTime = datetime.now()
                        runtime = endTime - startTime
                        print("Runtime: ", runtime.seconds / 60, " min")
                        same_score = model.score(features_train_val, target_train_val)  # Calculating accuracy in the same year

                        train_predict = model.predict(features_train)
                        n_predict = model.predict(valid_features)
                        score = model.score(valid_features, valid_targets)
                        # print(valid_target.shape)
                        # print(n_predict.shape)
                        obversed = np.squeeze(valid_targets)
                        print(obversed.shape)
                        accuracy = np.corrcoef(n_predict, obversed)[0, 1]
                        mse = mean_squared_error(obversed, n_predict)
                        print("The accuracy for {} in RM is: {}".format(trait, accuracy))
                        print("The mse for {} in RM is: {}".format(trait, mse))
                        print("The variance for predicted {} is: ".format(trait, np.var(n_predict)))
                        print("A bite of output:")
                        print("observe: ", obversed[:50])
                        print("predicted: ", n_predict[:50])
                        save_df = pd.DataFrame({"obv": obversed, "pred": n_predict})
                        save_df.to_csv("~/saved_outcomes.csv", sep="\t")

                        avg_acc.append(accuracy)
                        avg_score.append(score)
                        acg_same_score.append(same_score)
                        avg_mse.append(mse)
                        avg_runtime.append(runtime)
                        if accuracy > best_model[0]:
                            best_model = [accuracy, model]
                        r += 1
                        records.append([trait, "2013-15", "2017", hp_set[0],hp_set[1], same_score, score, accuracy, mse, region,runtime.seconds / 60])
                    if self.save is True:
                        saved_rf_model_fn = "{}{}_{}_{}_model.json".format(model_path, trait, self.method, region)
                        pickle.dump(best_model[1], open(saved_rf_model_fn, "wb"))
                    accs.append([trait, "2013-15", "2017", hp_set[0],hp_set[1], np.mean(acg_same_score), np.mean(avg_score),
                                 np.mean(avg_acc), np.mean(avg_mse),region,np.mean(avg_runtime).seconds / 60])
            check_usage()
            record_train_results(records, record_cols, method=self.method, path=record_path)
            record_train_results(accs, record_cols, self.method, path=record_path, extra="_summary")

        #record_train_results(accs, cols=record_cols, method="RM", path="~", para="max_features")
        #record_train_results(records, cols=record_cols, method="RM", path="~", para="max_feature_raw")



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
                accuracy_records = [0, None]
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
                            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE)
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

                            if accuracy_future > accuracy_records[0]:
                                print("Find a better model: ",accuracy_future)
                                accuracy_records = [accuracy_future,model]

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
                if self.save is True:
                    """
                    Save the model with the best accuracy in a h-parameter queue.
                    """
                    saved_json = accuracy_records[1].to_json()
                    with open("{}{}_{}_{}_model.json".format(model_path, trait, self.method, setting),
                              "w") as file:
                        file.write(saved_json)
                    accuracy_records[1].save_weights(
                        "{}{}_{}_{}_model.json.h5".format(model_path, trait, self.method, setting))
            check_usage()
        record_train_results(results, record_columns, method=self.method, path=record_path)
        record_train_results(record_summary, record_columns, self.method, path=record_path, extra="_summary")






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
    if config["BASIC"]["method"] in CNNs or  config["BASIC"]["method"] == "MLP":
        composer.trainning(model_path=model_path,record_path=record_path)
    elif config["BASIC"]["method"] == "RF":
        composer.make_forest(model_path=model_path,record_path=record_path)

if __name__ == "__main__":
    main()
