from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from Functions import *
import configparser
from GSModel import RM

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
    geno_data = decoding(geno_data)
    print(geno_data.columns)
    train_year = [2013,2014,2015]
    valid_year = [2017]
    filtered_data = read_pipes(geno_data,pheno_data,[2013,2014,2015,2017])


    train = filtered_data.query('Series in @train_year')
    valid = filtered_data.query('Series in @valid_year')

    print(train.info())
    print(train.Series.unique())
    print(valid.info())
    print(valid.Series.unique())
    #print(train.iloc[:,2].unique())


    traits = ["TCHBlup","CCSBlup","FibreBlup"]
    for trait in traits:
        model = RM()
        in_train = train.dropna(subset=[trait],axis=0)
        in_valid = valid.dropna(subset=[trait],axis=0)
        print(in_train.columns)
        train_target = in_train[[trait]]
        valid_target = in_valid[[trait]]
        print(valid_target)

        dropout = ["TCHBlup", "CCSBlup", "FibreBlup", "Region", 'Trial', 'Crop', 'Clone', 'sample']

        in_train.drop(dropout, axis=1, inplace=True)
        in_valid.drop(dropout,axis=1,inplace=True)
        print(train.columns)
        model.fit(in_train, np.array(train_target))

        n_predict = model.predict(in_valid)
        print(model.score(in_valid,valid_target.values))
        print(valid_target.shape)
        print(n_predict.shape)
        accuracy = np.corrcoef(n_predict, np.reshape(valid_target,(1,)))[0, 1]

        print(accuracy)

if __name__ == "__main__":
    main()
