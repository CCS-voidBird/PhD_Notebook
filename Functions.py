import pandas as pd
import numpy as np
import configparser

"""
This python file is for building functions that can associate with main model;
    1. add a function that can read/select/transform geno/pheno data from the origin geno/pheno files rather than 
        regenerating subgroups (e.g. 2016_TCHBlup_all.csv)
        
    2. Move/merge plot/model_save functions to this file.
"""


def mid_merge(x,genos):

    merged = (genos
           .query('sample in @x.Clone')
           .pipe((pd.merge,'right'),left=x,left_on="Clone",right_on="sample"))

    return merged

def read_pipes(genotype, phenotypes, years):
    """
    :param genotype: an overall genotype dataframe contains all clones; Format: raw = records, col = QTL names
    :param phenotypes: an overall phenotype dataframe conatins all traits and non-genetic features.
    :param years: selected years including training years and valid years => maybe contain other features in the future
    :return: training set and valid set that contain trait and genotypes.
    """

    #selected_phenos = phenotypes.iloc[phenotypes.Year in years]
    #selected_phenos = phenotypes.query('Series in @years')
    #selected_genos = genotype.query('sample in @selected_phenos.Clone')
    #selected_genos = genotype.iloc[genotype.sample in selected_phenos.Clone.values]
    #merged_data = pd.merge(selected_phenos,selected_genos,left_on="Clone",right_on="sample")

    print(years)
    goal = (phenotypes
            .query('Series in @years')
            .pipe(mid_merge,genos=genotype)
            )

    return goal

def decoding(data):

    snps = {'TT':0,'AT':1,'AA':2,'--':0.01}
    data.replace(snps,inplace=True)
    data.fillna("0.01",inplace=True)

    return data

def load_data(args):
    """
    :param paths: a list(tuple) of paths contains geno/pheno data
    :return: In-memory geno/pheno data
    """
    genotype = pd.read_csv(args.genotype,sep="\t")
    print("Got genotype index: \n",genotype.columns)
    genotype.drop(genotype.columns[0], axis=1, inplace=True) #Drop the sampling index of the genotype

    phenotype = pd.read_csv(args.phenotype,sep="\t")

    all_selected_series = args.train.split("-") + args.valid.split("-")

    filtered_data = read_pipes(genotype,phenotype,all_selected_series)

    return filtered_data


def factor_extender(data,factors):
    """
    This function helps create a 2D matrix from the origin 1D dataset (non-genetic factors, SNPs)
    :param data: merged data waiting for extend non-genetic factors
    :param factors: a list of non-genetic factors
    :return: a 2D matrix which contains: a line of genotype with m SNPs ,n lines of non-genetic factors, each line contain m repeated factors.
    """

    n_factor = data[factors]
    print(n_factor.shape)
    data.drop(factors,inplace=True,axis=1)
    n_repeat = data.shape[1]
    extended_factors = np.expand_dims(n_factor,1).repeat(n_repeat,axis=1)
    print("factor dim: ",extended_factors.shape)

    extended_genos = np.expand_dims(data,1)
    print("geno dim: ",extended_genos.shape)

    try:
        final_data = np.dstack([extended_factors, data])
        #final_data = np.dstack([extended_factors,extended_genos])
    except:
        print("Use concatenate")
        final_data = np.concatenate([extended_factors, extended_genos],axis=1)

    return final_data

def main():
    print("start")
    config = configparser.ConfigParser()
    config.read("./MLP_parameters.ini")
    geno_data=None
    pheno_data = None
    try:
        geno_data = pd.read_csv(config["PATH"]["genotype"],sep="\t")   # pd.read_csv("../fitted_genos.csv",sep="\t")
        pheno_data = pd.read_csv(config["PATH"]["phenotype"],sep="\t")# pd.read_csv("../phenotypes.csv",sep="\t")
    except:
        try:
            print("Using backup path (for trouble shooting)")
            geno_data = pd.read_csv(config["BACKUP_PATH"]["genotype"],sep="\t")  # pd.read_csv("../fitted_genos.csv",sep="\t")
            pheno_data = pd.read_csv(config["BACKUP_PATH"]["phenotype"],sep="\t")  # pd.read_csv("../phenotypes.csv",sep="\t")
        except:
            print("No valid path found.")
            quit()
    geno_data.drop(geno_data.columns[0],axis=1,inplace=True)
    print(geno_data.columns)
    years=[x for x in range(2013,2016)]
    goal = read_pipes(geno_data,pheno_data,years)
    traits = goal[config["TRAIT"]["traits"].split("#")]
    goal.drop(["TCHBlup","CCSBlup","FibreBlup","sample","Region"],inplace=True,axis=1)

    print("Finish transforming")
    print(goal.info())
    print(goal.Series.unique())
    ext_goal = factor_extender(goal, ["Series"])
    print(ext_goal.shape)
    print(ext_goal[:,:,1])
    print("done")

if __name__ == "__main__":
    main()