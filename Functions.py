import pandas as pd


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



def main():
    print("start")
    geno_data = pd.read_csv("../fitted_genos.csv",sep="\t")
    geno_data.drop(geno_data.columns[0],axis=1,inplace=True)
    print(geno_data.columns)
    pheno_data = pd.read_csv("../phenotypes.csv",sep="\t")
    years=[x for x in range(2013,2016)]
    goal = read_pipes(geno_data,pheno_data,years)
    print(goal.info())
    print(goal.Series.unique())
    print("done")

if __name__ == "__main__":
    main()