import pandas as pd
import torch
import numpy as np
import os
import argparse
PATH = "E:/learning resource/PhD/sugarcane/"
INTRO = "Data Construction - Sugarcane"
#trait_path = "E:/learning resource/PhD/phenotypes_full.csv"
#genos_path = "E:/learning resource/PhD/geno_data1_full.csv"

class Sugarcane:

    def __init__(self):
        self.genotypes = None
        self.phenotypes = None
        self.envs = {}
        self.fields = []

    def feed(self,genos_path,phenotypes_path):
        """
        Read genotype, phenotypes from given path; set pseudo value to NA in geno array;
        :param genos_path: file path of geno array
        :param phenotypes_path: file path of traits
        """
        self.phenotypes = pd.read_csv(phenotypes_path, sep="\t")
        self.genotypes = pd.read_csv(genos_path, sep="\t").T
        self.genotypes[self.genotypes == "--"] = 0.01
        self.genotypes.index = self.genotypes.index.set_names("sample")
        self.genotypes = self.genotypes.reset_index()
        print(self.genotypes.shape)
        for cid in ["Series","Region"]:
            self.envs[cid] = pd.unique(self.phenotypes[cid])


    def select(self,field,sample_size=0):
        """

        :param field: a tuple or list contains Series, Region, Trait
        :param sample_size:
        :return:
        """
        if field is tuple or field is list:
            print(field)
            series,region,trait = field
            sub_pheno = self.phenotypes[self.phenotypes["Series"] == series and self.phenotypes["Region"] == region][["Clone",trait]]
            merge = pd.merge(sub_pheno,self.genotypes,left_on="Clone",right_on="sample")
            merge.dropna(axis=0,how="any",inplace=True)
            if sample_size != 0:
                dataset = merge.sample(sample_size)
            else:
                dataset = merge
            dataset.drop(["sample","Clone"],axis=0,inplace=True)
            csvname = PATH  + "_".join(field) + ".csv"
            dataset.to_csv(csvname,sep="\t")
            return field,dataset
        else:
            print("field should be a list/tuple!")

    def select_single(self,label,name,trait,sample_size=0):
        field = label
        if label and name and trait:
            sub_pheno = self.phenotypes[self.phenotypes[field] == name][["Clone",trait]]
            merge = pd.merge(sub_pheno,self.genotypes,left_on="Clone",right_on="sample")
            print(merge.shape)
            merge.dropna(axis=0,how="any",inplace=True)

            if sample_size != 0:
                dataset = merge.sample(sample_size)
            else:
                dataset = merge
            dataset.drop(["sample","Clone"],axis=1,inplace=True)
            csvname = PATH + "_".join([str(name),trait]) + ".csv"
            dataset.to_csv(csvname, sep="\t")
            return (field,name,trait),dataset
        else:
            print("!")

    def get_fields(self):
        for series in self.envs["Series"]:
            for region in self.envs["Region"]:
                for trait in ["CCSBlup", "TCHBlup", "FibreBlup"]:
                    self.fields.append((series, region, trait))

def split_sample(sugarcane_data):

    for year in [2013,2014,2015]:
        trainset = [field for field in sugarcane_data.fields if year in field]
    set2016 = [field for field in sugarcane_data.fields if "2016" in field]
    set2017 = [field for field in sugarcane_data.fields if "2017" in field]

    return trainset,set2016,set2017


def main():

    parser = argparse.ArgumentParser(description=INTRO)
    req_grp = parser.add_argument_group(title='Required')
    req_grp.add_argument('-g', '--geno', type=str, help="Input genotype file.", required=True)
    req_grp.add_argument('-t', '--trait', type=str, help="Input phenotype file.", required=True)
    req_grp.add_argument('-o', '--output', type=str, help="Input output dir.", required=True)
    req_grp.add_argument('-f', '--filter-blank', type=bool, help="filter NA values", default=True)
    req_grp.add_argument('-s', '--sample', type=str, help="number of sample", default="all")
    args = parser.parse_args()
    if args.output[0] == "/":
        locat = '/' + args.output.strip('/') + '/'
    else:
        locat = args.output.strip('/') + '/'
    geno_path = args.geno
    pheno_path = args.trait
    sugarcane_data = Sugarcane ()
    sugarcane_data.feed(geno_path,pheno_path)
    sugarcane_data.get_fields()
    for year in [2013,2014,2015,2016,2017]:
        for trait in ["CCSBlup", "TCHBlup", "FibreBlup"]:
            sugarcane_data.select_single("Series",year,trait)


    for year in ["2013","2014","2015"]:
        trainset = [field for field in sugarcane_data.fields if year in field]
    set2016 = [field for field in sugarcane_data.fields if "2016" in field]
    set2017 = [field for field in sugarcane_data.fields if "2017" in field]

    #samples = phenotypes["Clone"]
    """
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
    print("DONE")

    #torch.save(trYs, "{}/{}_TrainData_traits.pt".format(locat,args.sample))
    #torch.save(tsXgenes, "{}/{}_TrainData_genos.pt".format(locat,args.sample))

if __name__ == "__main__":
    main()