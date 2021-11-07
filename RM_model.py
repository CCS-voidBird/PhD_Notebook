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
    print(geno_data.columns)
    train_year = [2013,2014,2015]
    valid_year = [2017]
    filtered_data = decoding(read_pipes(geno_data,pheno_data,[2013,2014,2015,2017]))


    train = filtered_data.query('Series in @train_year')
    valid = filtered_data.query('Series in @valid_year')

    print(train.info())
    print(train.Series.unique())
    print(valid.info())
    print(valid.Series.unique())

    print(train.iloc[:,2].unique())

    model = RM()
    train_target = train.TCHBlup
    valid_target = valid.TCHBlup

    dropout = ["TCHBlup","Region",'Trial', 'Crop', 'Clone','sample']
    print(train.columns)

    history = model.fit(train.drop(dropout,axis=1),train_target)

    predict = history.predict(valid.drop(dropout,axis=1))

    accuracy = np.corrcoef(predict, valid_target)[0, 1]

    print(accuracy)

if __name__ == "__main__":
    main()
