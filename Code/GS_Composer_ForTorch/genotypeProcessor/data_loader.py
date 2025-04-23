import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from genotypeProcessor.Plink_Reader import read_transform_plink_files
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

def decoding(data):
    """
    :param data: genotyping data (AA, AT, TT)
    :return: numeric genotype data
    """
    try:
        snps = {'TT':0,'AT':1,'AA':2,'--':0}
        nas = {np.nan : 0}
        data.replace(nas, inplace=True)
        data.fillna("0", inplace=True)
        data.replace(snps,inplace=True)
    except:
        print("All the SNPs are already decoded, imputing missing SNPs with 0")
    print("Convert data to np.array float32")
    data = np.asarray(data).astype(np.float32)

    return data

class DataProcessor:
    """
    This class is used to dealing with training date for each epoch
    """

    def __init__(self, args):
        self._raw_data = {"GENO":pd.DataFrame(),"PHENO":pd.DataFrame(),"INDEX":pd.DataFrame(),"ANNOTATION":pd.DataFrame()}
        self.train_data = None
        self.train_info = pd.DataFrame()
        self.train_pheno = pd.DataFrame()
        self.valid_data = None
        self.valid_info = pd.DataFrame()
        self.valid_pheno = pd.DataFrame()
        self._info = {}
        self.batchSize = 0
        self.args = args
        self.get_data()
        self.sort_data()

    def get_data(self):
        
        self._raw_data["GENO"],self._raw_data["MAP"] = read_transform_plink_files(self.args.geno)

        self._raw_data["FAM"] = pd.read_table(self.args.geno + ".fam", sep='\s+',header=None)
        self._raw_data["PHENO"] = pd.read_table(self.args.pheno, sep='\s+',header=None)
        self._raw_data["INDEX"] = pd.read_table(self.args.index,sep='\s+',header=None) if self.args.index is not None else self._raw_data["FAM"].iloc[:,0:3]
        self._raw_data["INFO"] = self._raw_data["FAM"].iloc[:,0:6]  #Further using fam file instead.

        if self.args.vindex == 0:
            self._raw_data["INDEX"].iloc[:,-1] = 0 ##set all individuals as training set

        self._raw_data["ANNOTATION"] = pd.read_table(self.args.annotation,sep='\s+') if self.args.annotation is not None else None

        self._info["CROSS_VALIDATE"] = sorted(self._raw_data["INDEX"].iloc[:,-1].unique()) 

        ##Perform basic statistics
        self._info["MARKER_SIZE"] = self._raw_data["MAP"].shape[0]
        self._info["MAF"] = self._raw_data["GENO"].iloc[:,6:].apply(lambda x: np.mean(x)/self.args.ploidy,axis=0)
        print("MAF brief review:")
        print(self._info["MAF"].head(10))
        print(self._info["MAF"].shape)

        self.batchSize = self.args.batch

        #prapare phenotype columns for multiTrait models
        self.pheno_col_index = self.args.mpheno + 1 if self.args.mpheno != 0 else list(range(2,len(self._raw_data["PHENO"].columns)))
        self.args.NumTrait = len(self.pheno_col_index) if self.args.mpheno == 0 else 1


        print("Get genotype shape:",self._raw_data["GENO"].iloc[:,6:].shape)
        #print(self._raw_data["GENO"].iloc[:,6:].iloc[1:10,1:10])
        
        return

    def sort_data(self):
        """
        Sort raw data as plink format
        FID,IID,father,mother,sex,pheno --> fam
        Chromosome, Variant ID, position, base pair --> map
        """
        # sort GENO by first col with reference FAM
        print("Running data check")

        sample_reference = self._raw_data["FAM"].iloc[:,1]## Get fam IID as reference
        modelling_reference = self._raw_data["INDEX"].iloc[:,1]
        snp_reference = self._raw_data["MAP"].iloc[:,:2]
        print("Checking samples alignment...")

        
        if self._raw_data["GENO"].iloc[:,6:].shape[1] != snp_reference.shape[0]:
            print("SNPs are not in same length in ped file and map file")
            print("SNP length in ped file: ",self._raw_data["GENO"].iloc[:,6:].shape[1])
            print("SNP length in map file: ",snp_reference.shape[0])
            ##select ped records that their second column value are appeared in fam file

        ##Run check between Geno 1-6 and fam file, should be exactly the same
        if self._raw_data["GENO"].iloc[:,0:6].equals(self._raw_data["FAM"]) is False:
            print("FAM file is not aligned with GENO file, but the fam file will be used as reference and replace infomations from geno file. Please double-check.")
            #replace geno file with fam file
            self._raw_data["GENO"].iloc[:,0:6] = self._raw_data["FAM"]
            

        ##sort ped, pheno and index file by IID order from fam file
        self._raw_data["GENO"] = self._raw_data["GENO"].loc[self._raw_data["GENO"].iloc[:,1].isin(modelling_reference),:].reset_index(drop=True)
        #print(self._raw_data["GENO"].iloc[1:10,1:10])
        self._raw_data["PHENO"] = self._raw_data["PHENO"].loc[self._raw_data["PHENO"].iloc[:,1].isin(modelling_reference),:].reset_index(drop=True)
        #print(self._raw_data["INDEX"].loc[self._raw_data["INDEX"].iloc[:,1].isin(sample_reference),:])
        self._raw_data["INDEX"] = self._raw_data["INDEX"].loc[self._raw_data["INDEX"].iloc[:,1].isin(modelling_reference),:].reset_index(drop=True)
        print("SNPs are filtered by fam file.")
        #Check if samples are aligned with same order with fam file


        if self.args.annotation is not None and self._raw_data["ANNOTATION"].iloc[:,:1].equals(snp_reference) is False:
            print("SNPs in annotation file are not ordered by map file")
            #exit()
        print("Data check & sort passed.\n")

        ## data quality control and filter by MAF
        if self.args.maf > 0:
            print("Filtering SNPs by MAF...")
            print("Loaded SNPs: ",self._info["MARKER_SIZE"])
            print("Filtering SNPs with MAF lower than {} with ployidy {}.".format(self.args.maf,self.args.ploidy))
            ### Get column index of MAF >= self.args.maf 
            selected_marker_idx = self._info["MAF"][self._info["MAF"] >= self.args.maf].index
            #print(selected_marker_idx[0:10])
            self._raw_data["GENO"] = self._raw_data["GENO"].iloc[:,selected_marker_idx]
            self._raw_data["MAP"] = self._raw_data["MAP"].iloc[selected_marker_idx-6,:] #update map file
            self._info["MARKER_SIZE"] = self._raw_data["MAP"].shape[0]
            print("Filtered SNPs: ",self._info["MARKER_SIZE"],"\n")

    def get_cross_validate(self):
        return self._info["CROSS_VALIDATE"]

    def prepare_training(self,train_index:list,valid_index):

        ##prepare training data, extract removal index
        removal = np.where(self._raw_data["PHENO"].iloc[:, self.pheno_col_index].isin([None,"NA",np.nan]))[0].tolist()

        print("Overall population: {}".format(len(self._raw_data["INDEX"].index)))
        print("{} individuals need to be removed due to the miss phenotype".format(len(removal)))
        train_mask = [x for x in np.where(self._raw_data["INDEX"].iloc[:, -1].isin(train_index))[0].tolist() if x not in removal]
        valid_mask = [x for x in np.where(self._raw_data["INDEX"].iloc[:, -1].isin(valid_index))[0].tolist() if x not in removal]
        print("Filtered population: {}".format(len(train_mask)+len(valid_mask)))

        print("Training set: {}".format(len(train_mask)))
        print("source shape: {}".format(self._raw_data["GENO"].shape))

        self.train_data = self._raw_data["GENO"].iloc[train_mask, 6:] * self._info["MAF"] if self.args.mafm is True else self._raw_data["GENO"].iloc[train_mask, 6:]
        self.valid_data = self._raw_data["GENO"].iloc[valid_mask, 6:] * self._info["MAF"] if self.args.mafm is True else self._raw_data["GENO"].iloc[valid_mask, 6:]
        self.train_pheno = self._raw_data["PHENO"].iloc[train_mask, self.pheno_col_index]


        self.mean_pheno = np.mean(self.train_pheno)
        if self.args.mean is not True or self.args.data_type == "ordinal":
            self.mean_pheno = 0


        self.train_pheno = self.train_pheno - self.mean_pheno
        self.valid_pheno = self._raw_data["PHENO"].iloc[valid_mask, self.pheno_col_index]
        self.train_pheno = np.asarray(self.train_pheno).astype(np.float32)
        self.valid_pheno = np.asarray(self.valid_pheno).astype(np.float32)

        if self.args.data_type == "ordinal":
            self.args.classes = np.max(self.train_pheno) + 1

            try:
                self.args.classes = int(max(np.max(self.train_pheno),np.max(self.valid_pheno)))

            except:
                print("Using backup function inherited from keras source code. "
                      "You may need to use code 'pip install tf-nightly' to install the actual module.")
                self.args.classes = int(max(np.max(self.train_pheno), np.max(self.valid_pheno)))
                print(self.args.classes)

        return Dataset_train(self.train_data, self.train_pheno, tag="train", args=self.args), Dataset_valid(self.valid_data, self.valid_pheno, tag="valid", args=self.args)
    
    def make_dataset(self):

        return Dataset(
            self.sort_data["GENO"].iloc[:, 6:] * self._info["MAF"] if self.args.mafm is True else self.sort_data["GENO"].iloc[:, 6:],
            self.sort_data["PHENO"].iloc[:, self.pheno_col_index],tag="all",args=self.args
        )

class Dataset_train(Dataset):

    def __init__(self, data, pheno, tag="train", args=None):
        self.data = data
        self.pheno = pheno
        self.args = args
        tag = tag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_X = self.data.iloc[idx, :].values
        feature_y = self.pheno[idx]

        return feature_X, feature_y
    
    def get_shape(self):
        #return the shape of the data without the first dimension
        return self.data.shape[1:]

class Dataset_valid(Dataset):

    def __init__(self, data, pheno, tag="valid", args=None):
        self.data = data
        self.pheno = pheno
        self.args = args
        tag = tag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_X = self.data.iloc[idx, :].values
        feature_y = self.pheno[idx]

        return feature_X, feature_y