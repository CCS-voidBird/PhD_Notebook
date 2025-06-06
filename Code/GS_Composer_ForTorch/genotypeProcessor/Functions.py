import pandas as pd
import numpy as np
import configparser
import argparse
import platform
import psutil
import os

"""
This python file is for building functions that can associate with main model;
    1. add a function that can read/select/transform geno/pheno data from the origin geno/pheno files rather than 
        regenerating subgroups (e.g. 2016_TCHBlup_all.csv)
        
    2. Move/merge plot/model_save functions to this file.
"""
"""
config = configparser.ConfigParser()
if platform.system().lower() == "windows":
    config.read("./MLP_parameters.ini")
else:
    config.read("/clusterdata/uqcche32/MLP_parameters.ini")
"""

def calculate_correlation_for_traits(y_pred,y_true,numTraits=1):
    """
    Calculate the correlation between predicted value and true value for each trait, nan value woule be missed by pairs.
    :param y_true: true value
    :param y_pred: predicted value
    :param numTraits: number of traits
    :return: a list of correlation value for each trait
    """
    corrs = []
    for i in range(numTraits):
        #Get the true and predicted value for each trait
        y_true_trait = y_true[:,] if numTraits == 1 else y_true[:,i]
        y_pred_trait = y_pred[:,] if numTraits == 1 else y_pred[:,i]
        #reshape y_true and y_pred to 1D array
        y_true_trait = np.reshape(y_true_trait,(-1,))
        y_pred_trait = np.reshape(y_pred_trait,(-1,))
        #construct a mask to filter nan value
        mask = ~np.isnan(y_true_trait) & ~np.isnan(y_pred_trait)
        #calculate the correlation between true and predicted value
        corr = np.corrcoef(y_true_trait[mask],y_pred_trait[mask])[0,1]
        corrs.append(corr)
        
    return corrs

def read_transform_plink_files(geno_path):

    print("Reading geno files from: ",geno_path)
    """
    convert plink format to pandas dataframe
    """
    full_ped_name = geno_path + ".ped"
    full_map_name = geno_path + ".map"
    #full_fam_name = geno_path + ".fam"

    map_data = pd.read_csv(full_map_name,sep=r"\s+",header=None)
    print("Read {} markers from map file.".format(map_data.shape[0]))
    #check map file infos, if V3 == V4, then raise a warning, then let V4 = 0
    if map_data.iloc[:,2].equals(map_data.iloc[:,3]):
        #print("V3 is same with V4, set V4 to 0")
        map_data.iloc[:,3] = 0

    marker_list = map_data.iloc[:,1].values
    #sample_list = fam_data.iloc[:,1].values
    #ref_allele = map_data.iloc[:,3].values
    

    ped = pd.read_csv(full_ped_name,sep=r"\s+",header=None)
    
    #ped = pd.read_csv(full_ped_name,sep="\t",header=None)
    ped_1_field = ped.iloc[:,0:6]
    ped_2_field = ped.iloc[:,6:] 
    ped_2_field.replace(["--",-9,"NA"],np.nan,inplace=True)
    #Identify minor allele with lower allele frequyency
    print("Sorting minor allele.......", end="")
    ped_2_field_minor_allele = ped_2_field.apply(lambda x: x.value_counts().idxmin(),axis=0)
    ped_2_field_minor_allele.replace("0","T",inplace=True)
    ##Reindex the minor allele from 0 to the last marker
    ped_2_field_minor_allele.index = ped_2_field_minor_allele.index-6
    print("DONE")#,ped_2_field_minor_allele.head())
    #ped_2_field_minor_allele = ped_2_field_minor_allele.replace("0","T")

    if ped_2_field.shape[1] == len(marker_list):
        print("The number of markers are same with map file presented.")
        ped_2_field.columns = marker_list
        #check if values in variant_allele is any integer or not
        #print(ped_2_field.dtypes)
        if ped_2_field.dtypes.unique().all() == "int64":
            print("Variant allele is integer, no need to transform.")
        else:
            # count maximum string length across entire dataframe
            ped_2_field.replace(" ","",inplace=True)
            ploidy = ped_2_field.values.astype(str).max(axis=0).max()
            print("Detected ploidy: ",ploidy)

            # for each individual x and each marker i, count the refernence allele number from the ped_2_field_minor_allele[i]
            for i, col in enumerate(ped_2_field.columns):
                target_char = ped_2_field_minor_allele[i]
                ped_2_field[col] = ped_2_field[col].apply(lambda x: ploidy - x.count(target_char))

    elif ped_2_field.shape[1] % len(marker_list) == 0 and ped_2_field.shape[1] > len(marker_list):
        ploidy = ped_2_field.shape[1] // len(marker_list)
        print("Detected ploidy: ",ploidy)
        
        #reshape the ped genos from (N,marker*ploidy) to (N,marker,ploidy)
        ped_2_field_reshaped = np.reshape(ped_2_field.values,(ped_2_field.shape[0],len(marker_list),ploidy))

        if ped_2_field.dtypes.unique().all() == "int64":
            print("Variant allele is integer, no need to transform.")
        else:
            
            print("Variant allele is string, transforming...Now convert allele genotypes to allele counts.")
            #print(ped_2_field_reshaped[0,0,:])
            for i in range(len(marker_list)):
                ped_2_field_reshaped[:,i,:] = np.where(ped_2_field_reshaped[:,i,:] == ped_2_field_minor_allele[i], 0, 1)

        #sum the ploidy to get the final geno data
        ped_2_field = np.sum(ped_2_field_reshaped,axis=2)

    else:
        print("Cannot match the marker number with the genotype data.")
        exit()
    # combine the ped_1_field and ped_2_field
    ped_2_field_merged = pd.DataFrame(np.concatenate([ped_1_field,ped_2_field],axis=1))
    print("Finish transforming genotype data.")
    #print(ped_2_field_merged.head())
    return ped_2_field_merged, map_data



def select_subset(config,geno,pheno,select_by):
    traits = config["BASIC"]["traits"]
    train_year = config["BASIC"]["train"]
    valid_year = config["BASIC"]["valid"]
    non_genetic_factors = [x for x in pheno.columns if x not in traits]
    print("Detected non-genetic factors from phenotype file: ", non_genetic_factors)
    filtered_data = read_pipes(geno, pheno, train_year + valid_year)
    dropout = [x for x in non_genetic_factors if
               x not in ["Region", "Series"]] + ["Sample"]  # config["BASIC"]["drop"].split("#") + ['Sample']
    print("Removing useless non-genetic factors: {}".format(dropout))
    filtered_data.drop(dropout, axis=1, inplace=True)

    select_data = filtered_data.query('Region == @select_by').drop(["Region"],axis=1)

    return select_data

def get_overlapping(data,train,valid):
    train_clones = data.query("Series in @train").Sample.unique()
    valid_clones = data.query("Series in @valid").Sample.unique()
    overlap_list = np.intersect1d(train_clones,valid_clones)
    print("Find {} overlapped clones".format(len(overlap_list)))
    print("The overlapped clones in train~valid set:")
    print(str(overlap_list))

    return overlap_list


def remove_clones(data:pd.DataFrame,remove_list):
    l = data.shape[0]
    clean_data = data.query("Sample not in @remove_list")
    print("Removed {} records from data.".format(l-clean_data.shape[0]))

    return clean_data

def mid_merge(x,genos):

    merged = (genos
           .query('Sample in @x.Clone')
           .pipe((pd.merge,'right'),left=x,left_on="Clone",right_on="Sample"))

    return merged

def create_subset(data:pd.DataFrame,factor_value,factor_name="Region"):
    if factor_name == "Region":
        subset = data.query('Region is @factor_value')
        subset.drop(factor_name,axis=1)
    else:
        print("currently cannot resolve domain without region tag./")
        return pd.DataFrame()
    return subset

def to_ordinal(y, num_classes=None, dtype="float32"):
    """
    A backup function for using keras to_ordinal function
    """

    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    range_values = np.arange(num_classes - 1)
    range_values = np.tile(np.expand_dims(range_values, 0), [n, 1])
    ordinal = np.zeros((n, num_classes - 1), dtype=dtype)
    ordinal[range_values < np.expand_dims(y, -1)] = 1
    output_shape = input_shape + (num_classes - 1,)
    ordinal = np.reshape(ordinal, output_shape)
    return ordinal

def combi(seq):
    """
    :param seq: hyper-parameter settings from config file
    :return: a list of hp sets
    """
    if not seq:
        yield []
    else:
        for element in seq[0]:
            for rest in combi(seq[1:]):
                yield [element] + rest

def snp_extend(genotypes):

    print("START transfer")
    n_sample = genotypes.shape[0]
    l = int(np.ceil(np.sqrt(genotypes.shape[1])))
    extend = l**2 - genotypes.shape[1]
    print(extend)
    snps = np.pad(np.array(genotypes),((0,0),(0,extend)),'constant',constant_values = (0.01,0.01))

    snps_2d = np.reshape(snps,(n_sample,l,l))
    print(snps_2d.shape)

    return snps_2d


def read_pipes(genotype, phenotypes, years,blue=None):
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

    print("Got selected years:",years)
    if blue is None:
        goal = (phenotypes
                .query('Series in @years')
                .pipe(mid_merge,genos=genotype)
                )
    else:
        samples = phenotypes.query('Series in @years').Clone.unique()
        goal = (blue
                .query('Clone in @samples')
                .pipe(mid_merge,genos=genotype))

    return goal

def check_usage():
    info = psutil.virtual_memory()
    print("Resource check:")
    print("Total memory: %.4f GB" % (info.total/1024/1024/1024))
    cpu_memory = psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024
    print("Currently using memory: %.4f GB" % cpu_memory)
    print("Ratio of used memory: %.4f " % (info.percent),"%")
    print("Number of CPU node: ",psutil.cpu_count())
    print("CPU usage: ",psutil.cpu_percent(interval=1))



def check_gpu_usage():
    #print overall gpu memory usage
    print("GPU usage:")
    os.system("nvidia-smi")





def decoding(data):
    """
    :param data: genotyping data (AA, AT, TT)
    :return: numeric genotype data
    """
    try:
        snps = {'TT':0,'AT':1,'AA':2,'--':0.01}
        nas = {np.nan : 0.01}
        data.replace(nas, inplace=True)
        data.fillna("0.01", inplace=True)
        data.replace(snps,inplace=True)
    except:
        print("All the SNPs are already decoded, imputing missing SNPs with 0.01")
    print("Convert data to np.array float32")
    data = np.asarray(data).astype(np.float32)

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

def get_years(years):

    if len(years) > 1:
        start = years.split("-")[0]
        end = years.split("-")[-1]
        years = [x for x in range(int(start),int(end)+1)]

    return years

def data_filter(data,filter_cols):

    pass

def record_train_results(results:list,cols,method,path = ".",para = "default",extra=''):
    """
    This function records training performance to csv file.
    :param results: a list contain performance data
    :param cols: columns for recording requirements
    :param method: training model
    :param path: output Path
    :param para: specific note for training
    """
    if isinstance(results,list) is not True:
        print("The training results should be a dictionary.")
    if len(cols) != len(results[0]):
        print("Cannot match records with columns, printing raw records instead")
        print(cols)
        print(results)
    record = pd.DataFrame(results,columns=cols)
    record.to_csv("{}/{}_train_record_by_{}{}.csv".format(path,method,para,extra),sep="\t")
    print("Result:")
    print(record)

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

def get_args():
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='Required')
    #req_grp.add_argument('-p', '--path', type=str, help="Input path.", required=True)
    #req_grp.add_argument('-1', '--train', type=str, help="Input train year.", required=True)
    #req_grp.add_argument('-2', '--valid', type=str, help="Input valid year.", required=True)
    req_grp.add_argument('-m', '--method', type=str, help="Select training method (CNN/MLP).", default="CNN")
    req_grp.add_argument('-o', '--output', type=str, help="Input output dir.")
    req_grp.add_argument('-s', '--sample', type=str, help="number of sample", default="all")
    req_grp.add_argument('-a', '--region', type=bool, help="add regions (T/F)", default=False)
    req_grp.add_argument('-r', '--round', type=int, help="training round.", default=20)
    req_grp.add_argument('-epo', '--epoch', type=int, help="training epoch.", default=50)
    req_grp.add_argument('-oh', '--onehot', type=int, help="One Hot encoder switch", default=1)
    req_grp.add_argument('-opt', '--optimizer', type=str, help='select optimizer: Adam, SGD, rmsprop',
                         default="rmsprop")
    req_grp.add_argument('-plot', '--plot', type=bool, help="show plot?",
                         default=False)
    req_grp.add_argument('-sli', '--silence', type=bool, help="silent mode",
                         default=True)
    req_grp.add_argument('-loss', '--loss', type=int, help="The target loss",
                         default=10)
    req_grp.add_argument('-save', '--save', type=bool, help="save model True/False",
                         default=False)
    req_grp.add_argument('-config', '--config', type=str, help='config file path, default: ~/MLP_parameters.ini',
                         default="~/MLP_parameters.ini",required=True)
    args = parser.parse_args()

    return args



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
