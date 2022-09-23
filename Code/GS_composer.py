
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
    req_grp.add_argument('-m', '--model', type=str, help="Select training model.", required=True)
    req_grp.add_argument('-l', '--load', type=str, help="load model from file.", default=None)
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
        self._raw_data = {"GENO":pd.DataFrame(),"PHENO":pd.DataFrame(),"INDEX":pd.DataFrame(),"ANNOTATION":pd.DataFrame()}
        self.train_data = None
        self.train_info = pd.DataFrame()
        self.train_pheno = pd.DataFrame()
        self.valid_data = None
        self.valid_info = pd.DataFrame()
        self.valid_pheno = pd.DataFrame()
        self._model = {"INIT_MODEL":Model(),"TRAINED_MODEL":None}
        self._info = {}
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

        self._raw_data["GENO"] = pd.read_table(args.ped,sep="\t",header=None)
        self._raw_data["PHENO"] = pd.read_table(args.pheno, sep="\t", header=None)
        self._raw_data["INDEX"] = pd.read_table(args.index,sep="\t", header=None)
        index_col = self._raw_data["INDEX"].shape[1]-1
        self._info["CROSS_VALIDATE"] = sorted(self._raw_data["INDEX"][index_col].unique())

        print(self._raw_data["INDEX"][index_col].value_counts().sort_values())

        self._raw_data["INFO"] = self._raw_data["GENO"].iloc[:,0:6]  #Further using fam file instead.

        print("Get genotype shape:",self._raw_data["GENO"].iloc[:,6:].shape)
        print(self._raw_data["GENO"].iloc[:,6:].iloc[1:10,1:10])

        """
        #self.train_data = self._raw_data["GENO"].query('X.1 in @train_sample').query('X.1 not in @remove_list').iloc[:,6:]
        #self.valid_data = self._raw_data["GENO"].query('X.1 in @valid_sample').iloc[:,6:]
        #print(self.train_data.shape)
        #print(self.valid_data.shape)

        #self.train_data = filtered_data.query('Series in @train_year').query('Sample not in @remove_list').drop(["Series","Sample"], axis=1)
        #self.valid_data = filtered_data.query('Series in @valid_year').drop(["Series","Sample"], axis=1)
        """
        return

    def prepare_model(self):
        # create a Model object

        if self.args.load is not None:
            self._model["INIT_MODEL"].load_model(self.args.load)
        else:
            self._model["INIT_MODEL"].init_model()

        # get data requirements - dimension, annotations, etc
        return

    def prepare_training(self,train_index:list,valid_index:list):

        train_mask = np.where(self._raw_data["INFO"].iloc[:, -1] in train_index)
        valid_mask = np.where(self._raw_data["INFO"].iloc[:, -1] in valid_index)

        self.train_data = self._raw_data["GENO"].iloc[train_mask, 6:]
        self.valid_data = self._raw_data["GENO"].iloc[valid_mask, 6:]

        self.train_pheno = self._raw_data["PHENO"].iloc[train_mask,self.args.pheno + 1]
        self.valid_pheno = self._raw_data["PHENO"].iloc[valid_mask, self.args.pheno + 1]

        label_encoder = LabelEncoder()

        self.prepare_model()
        self.train_data = self._model["INIT_MODEL"].data_transform(self.train_data) ## The raw data to transform include geno, pheno, annotations
        self.valid_data = self._model["INIT_MODEL"].data_transform(self.valid_data)

        self.train_data = np.asarray(self.train_data).astype(np.float32)
        self.train_pheno = np.asarray(self.train_pheno).astype(np.float32)
        self.valid_data = np.asarray(self.valid_data).astype(np.float32)
        self.valid_pheno = np.asarray(self.valid_pheno).astype(np.float32)

        return

    def compose(self):

        print("Train status:")
        print("Epochs: ",self.args.epoch)
        print("Repeat(Round): ",self.args.round)

        for round in self.args.round:

            n_features = self.train_data.shape[1:]
            self._model["TRAINED_MODEL"] = self._model["INIT_MODEL"].modelling(data_shape=n_features)

        pass

class Model(ML_composer):

    def __init__(self):

        super().__init__()

        self._init_model = None
        self._data_requirements = None
        self.modelling = None
        self.data_transform = None

    def init_model(self):

        self._init_model = MODELS[self.args.model]()
        self.data_transform = self._init_model.data_transform
        self.modelling = MODELS[self.args.model].model

        pass

    def load_model(self,path):
        #self._init_model = load(path)
        pass





