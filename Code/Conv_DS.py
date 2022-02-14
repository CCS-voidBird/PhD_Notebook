
from Functions import *
import pandas as pd
import keras.metrics

import numpy as np
import matplotlib.pyplot as plt
import platform
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
trait = "TCHBlup"
sorted_phenos = pheno_data.sort_values(by=["TCHBlup"],ascending=False)[:100]
sorted_samples = sorted_phenos.Clone.tolist()
sorted_traits = sorted_phenos[trait].tolist()

# sort genotypes by Sample and sorted_phenos_clone
# give a list of sample, sort geno_data by given index
# return a list of genotypes
def sort_geno(sorted_samples, geno_data):
    geno_data = geno_data.sort_values(by=["Sample"], ascending=True)
    geno_data = geno_data.set_index("Sample")
    geno_data = geno_data.loc[sorted_samples]
    geno_data = geno_data.values
    return geno_data
geno_data = geno_data.set_index("Sample")
SNPs = geno_data.loc[sorted_samples]
#SNPs = sorted_genos.drop(["Sample"], axis=1)

print(SNPs.shape)

path = "E:/learning resource/PhD/HPC_Results/CNN/backup/2013-2015_vs_2017/models/"
region = "all"
method = "CNN"
model_path = path + "{}_{}_{}_model.json".format(trait,method,region)
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
sample_convs = []
activation_model = models.Model(inputs=trained_model.input, outputs=layer_outputs)
for layer in trained_model.layers:
    print(layer)
activation_model.summary()
X = np.expand_dims(SNPs,axis=2)
activations = activation_model.predict(X)
conv = 2
layer_activation = activations[conv]
print(layer_activation.shape)


"""
# Plot conv feature maps by channels
for c in range(layer_activation.shape[-1]):
    plt.matshow(layer_activation[:,:,c], cmap='viridis')
    plt.savefig("E:/learning resource/PhD/HPC_Results/CNN/backup/activation_maps/{}_{}_{}_activation_map_{}.png".format(trait,method,region,c))
"""

#reshape the feature maps by channels
img_data = np.expand_dims(layer_activation,axis=0)
print(img_data.shape)
o,n,m,c = img_data.shape
reshape_img_data = img_data.reshape((o,c,m,n))
reshaped_img = np.where((0 < reshape_img_data), reshape_img_data, 0)
snp_importance = reshaped_img.sum(axis=1)
snp_importance = np.reshape(snp_importance,[n,m])
print(snp_importance.shape)
fig_imp = plt.figure()
axe_imp = fig_imp.add_subplot(111)

sample_index= 0
img_imp = axe_imp.bar(x=range(m),height=snp_importance[sample_index])
axe_imp.set_xlabel("SNP")
axe_imp.set_ylabel("Importance")
plt.savefig("E:/learning resource/PhD/HPC_Results/CNN/backup/importance_maps/{}_{}_{}_conv{}_importance.png".format(trait,method,region,conv))

"""
for i in range(X.shape[0]//100):
    end = i*100+100
    if i*100 > X.shape[0]:
        end = X.shape[0]
    activations = activation_model.predict(X[i*100:end])
    for l in range(len(activations)):
        layer_activation = activations[l]
        print(layer_activation.shape)
        sample_convs[l].extend(layer_activation)
"""