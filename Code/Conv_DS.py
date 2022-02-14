
from Functions import *
from GSModel import *
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import MaxPooling1D, Flatten, Dense, Conv1D,MaxPooling2D, Conv2D
from keras.layers import Dropout
import keras.metrics

import numpy as np

import platform
from datetime import datetime
from sklearn.metrics import mean_squared_error
import configparser
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
#Load genomic data and phenotypes
config_path = "./DS_config.ini"
config = configparser.ConfigParser()
if platform.system().lower() == "windows":
    print(config_path)
    config.read(config_path)
else:
    config.read(config_path)

try:
    geno_data = pd.read_csv(config["PATH"]["genotype"], sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
    pheno_data = pd.read_csv(config["PATH"]["phenotype"], sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
except:
    try:
        print("Using backup path (for trouble shooting)")
        print(config["BACKUP_PATH"]["genotype"])
        geno_data = pd.read_csv(config["BACKUP_PATH"]["genotype"],
                                sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
        pheno_data = pd.read_csv(config["BACKUP_PATH"]["phenotype"],
                                    sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
    except:
        print("No valid path found.")
        exit()
geno_data = decoding(geno_data)
samples = geno_data.Sample
SNPs = geno_data.drop(["Sample"], axis=1)

path = "E:/learning resource/PhD/HPC_Results/CNN/backup/2013-2015_vs_2017/models/"
trait_name = "TCHBlup"
region = "all"
method = "CNN"
model_path = path + "{}_{}_{}_model.json".format(trait_name,method,region)
weights_path = model_path + ".h5"
print(weights_path)
#region_index = "all"
#trait = "TCHBlup"
f_json = open(model_path,"r")
model_json = f_json.read()
f_json.close()
trained_model = keras.models.model_from_json(model_json)
trained_model.load_weights(weights_path)

from keras import models

layer_outputs = [layer.output for layer in trained_model.layers[:3]]
sample_convs = [[],[],[],[]]
activation_model = models.Model(inputs=trained_model.input, outputs=layer_outputs)
for layer in trained_model.layers:
    print(layer)
activation_model.summary()
X = np.expand_dims(SNPs,axis=2)
for i in range(X.shape[0]//100):
    end = i*100+100
    if i*100 > X.shape[0]:
        end = X.shape[0]
    activations = activation_model.predict(X[i*100:end])
    for l in range(len(activations)):
        layer_activation = activations[l]
        print(layer_activation.shape)
        sample_convs[l].extend(layer_activation)