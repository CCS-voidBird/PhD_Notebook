import pandas as pd
import torch
import numpy as np
import os
import argparse

INTRO = "Data Construction - Sugarcane"
#trait_path = "E:/learning resource/PhD/phenotypes_full.csv"
#genos_path = "E:/learning resource/PhD/geno_data1_full.csv"



def main():

    parser = argparse.ArgumentParser(description=INTRO)
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument('-g', '--geno', type=str, help="Input genotype file.", required=True)
    req_grp.add_argument('-t', '--trait', type=str, help="Input phenotype file.", required=True)
    req_grp.add_argument('-o', '--output', type=str, help="Input output dir.", required=True)
    req_grp.add_argument('filter', '--filter-blank', type=bool, help="filter NA values", default=True)
    req_grp.add_argument('-s', '--sample', type=str, help="number of sample", default="all")
    args = parser.parse_args()
    mode = args.filter
    if args.output[0] == "/":
        locat = '/' + args.output.strip('/') + '/'
    else:
        locat = args.output.strip('/') + '/'
    geno_path = args.geno
    pheno_path = args.trait

    phenotypes = pd.read_csv(pheno_path, sep="\t")
    genos = pd.read_csv(geno_path, sep="\t")
    genos[genos == "--"] = 0.01

    #samples = phenotypes["Clone"]

    arrays = genos.T
    arrays.dropna(axis=0, how="any", inplace=True)
    arrays.index = arrays.index.set_names("sample")
    arrays = arrays.reset_index()
    merged_data = pd.merge(phenotypes, arrays, left_on="Clone", right_on="sample")
    if args.sample == "all":
        size = merged_data.shape[0]
    else:
        size = int(args.sample)
    merged_data_small = merged_data.sample(size)
    traits = ["CCSBlup", "TCHBlup", "FibreBlup"]
    Ys = pd.DataFrame()

    for trait in traits:
        Y = merged_data_small[trait]
        Ys = pd.concat([Ys, Y], axis=1)

    # print(merged_data_small)
    Xgenes = merged_data_small.copy()
    drops = ["Series", "Region", "Trial", "Crop", "Clone", "sample", "CCSBlup", "TCHBlup", "FibreBlup"]
    # reduce memory usage
    for drop in drops:
        try:
            Xgenes.drop(drop, axis=1, inplace=True)
        except:
            print(drop, "has been dropped")

    npXgenes = np.array(Xgenes)
    npXgenes = npXgenes.astype(float)
    tsXgenes = torch.tensor(npXgenes)

    npY = np.array(Ys)
    trYs = torch.tensor(npY)

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

    torch.save(trYs, "{}/{}_TrainData_traits.pt".format(locat,args.sample))
    torch.save(tsXgenes, "{}/{}_TrainData_genos.pt".format(locat,args.sample))