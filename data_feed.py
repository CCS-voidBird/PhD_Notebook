import pandas as pd
import torch
import numpy as np

trait_path = "E:/learning resource/PhD/phenotypes_full.csv"
genos_path = "E:/learning resource/PhD/geno_data1_full.csv"

phenotypes = pd.read_csv(trait_path,sep="\t")
genos = pd.read_csv(genos_path,sep="\t")
genos[genos=="--"] = 0.01
#phenotypes[phenotypes == "NA"] = 0.01

samples = phenotypes["Clone"]

arrays = genos.T
#print(arrays)
arrays.dropna(axis=0,how="any",inplace=True)
arrays.index = arrays.index.set_names("sample")
arrays = arrays.reset_index()
merged_data = pd.merge(phenotypes,arrays,left_on="Clone",right_on="sample")
print(merged_data.head(10))
#nas = merged_data[merged_data == "NA"].index
#print(nas)
#merged_data = merged_data.drop(nas,axis=0)
#print(merged_data)
size = 2000
merged_data_small = merged_data.sample(size)
print(merged_data_small.shape)
traits = ["CCSBlup","TCHBlup","FibreBlup"]
Ys = pd.DataFrame()
for trait in traits:
    Y = merged_data_small[trait]
    Ys = pd.concat([Ys,Y],axis=1)


#print(merged_data_small)
Xgenes = merged_data_small.copy()
drops = ["Series","Region","Trial","Crop","Clone","sample","CCSBlup","TCHBlup","FibreBlup"]
for drop in drops:
    try:
        Xgenes.drop(drop,axis=1,inplace = True)
    except:
        print(drop,"has been dropped")

npXgenes = np.array(Xgenes)
npXgenes = npXgenes.astype(float)
tsXgenes = torch.tensor(npXgenes)

npY = np.array(Ys)
#npXg = np.array(Xgenes)
print(npY.shape)
trYs = torch.tensor(npY)

#print(merged_data_small)
"""
Xenv = pd.DataFrame()
envs = ["Series","Region","Trial","Crop"]
for env in envs:
    xenv = merged_data_small[env]
    Xenv = pd.concat([Xenv,xenv],axis=1)
print(Xenv.head(2))
npenv = np.array(Xenv)
print(npenv)
trXenv = torch.tensor(npenv)
"""
print("DONE")

torch.save(trYs,"../complete_TrainData_traits.pt")
torch.save(tsXgenes,"../complete_TrainData_genos.pt")