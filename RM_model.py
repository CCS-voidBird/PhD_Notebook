from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from Functions import *
import configparser
from GSModel import RM

def main():
    print("start")
    config = configparser.ConfigParser()
    #config.read("./MLP_parameters.ini")
    config.read("/clusterdata/uqcche32/MLP_parameters.ini")
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
            exit()
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
    accs = [] # pd.DataFrame(columns=["trait","trainSet","validSet","score","cov"])
    r = 0

    for trait in traits:
        avg_acc = []
        avg_score = []
        while r < 10:
            model = RM()
            in_train = train.dropna(subset=[trait], axis=0)
            in_valid = valid.dropna(subset=[trait], axis=0)
            # print(in_train.columns)
            train_target = in_train[[trait]]
            valid_target = in_valid[[trait]]

            dropout = ["TCHBlup", "CCSBlup", "FibreBlup", "Region", 'Trial', 'Crop', 'Clone', 'sample']

            in_train.drop(dropout, axis=1, inplace=True)
            in_valid.drop(dropout, axis=1, inplace=True)

            print(in_train.columns[:10])
            model.fit(in_train, np.array(train_target))

            n_predict = model.predict(in_valid)
            score = model.score(in_valid, valid_target.values)
            # print(valid_target.shape)
            # print(n_predict.shape)
            obversed = np.squeeze(valid_target)
            print(obversed.shape)
            accuracy = np.corrcoef(n_predict, obversed)[0, 1]

            print("The accuracy for {} in RM is: {}".format(trait, accuracy))
            print("A bite of output:")
            print("observe: ", obversed[:10])
            print("predicted: ", n_predict[:10])
            avg_acc.append(accuracy)
            avg_score.append(score)
            r += 1
        accs.append([trait, "2013-15", "2017", np.mean(avg_score), np.mean(avg_acc)])

    results = pd.DataFrame(accs,columns=["trait","trainSet","validSet","score","accuracy"])
    print("Result:")
    print(results)

if __name__ == "__main__":
    main()
