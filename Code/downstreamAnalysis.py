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

modelling = METHODS["CNN"]

model_path = "../../HPC_Results/CNN/TCHBlup_CNN_A_model.json"
f_json = open(model_path,"r")
model_json = f_json.read()
f_json.close()
trained_model = keras.models.model_from_json(model_json)


geno_data = pd.read_csv("../../genomic data/qc_genotypes",sep="\t")
pheno_data = pd.read_csv("../../phenotypes.csv",sep="\t")

for layer in trained_model.layers:
    weight = layer.get_weights()
    print(weight)

