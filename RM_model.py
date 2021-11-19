from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from Functions import *
import configparser
from GSModel import RM
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import platform
import joblib

def main():
    print("start")
    config = configparser.ConfigParser()
    if platform.system().lower() == "windows":
        config.read("./MLP_parameters.ini")
    else:
        config.read("/clusterdata/uqcche32/MLP_parameters.ini")

    geno_data=None
    pheno_data = None
    traits = config["BASIC"]["traits"].split("#")

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
    record_cols = ["trait", "trainSet", "validSet", "n_features", "test_score", "valid_score", "accuracy", "mse"]

    train = filtered_data.query('Series in @train_year')
    valid = filtered_data.query('Series in @valid_year')

    print(train.info())
    print(train.Series.unique())
    print(valid.info())
    print(valid.Series.unique())
    #print(train.iloc[:,2].unique())



    accs = [] # pd.DataFrame(columns=["trait","trainSet","validSet","score","cov"])
    records = []
    max_feature_list = [int(x) for x in config["RM"]["max_features"].split(",")]

    for n_features in max_feature_list:
        print("Now training by {} feature per tree.".format(n_features))
        for trait in traits:
            print(trait)
            avg_acc = []
            avg_score = []
            acg_same_score = []
            avg_mse = []
            r = 0
            while r < 10:
                model = RM(specific=True,n_features=n_features)

                """
                Drop rows that contain NaN in trait value.
                """
                in_train = train.dropna(subset=[trait], axis=0)
                in_valid = valid.dropna(subset=[trait], axis=0)
                # print(in_train.columns)
                train_target = np.squeeze(in_train[[trait]]).ravel()
                valid_target = np.squeeze(in_valid[[trait]]).ravel()

                dropout = ["TCHBlup", "CCSBlup", "FibreBlup", "Region", 'Trial', 'Crop', 'Clone', 'Series', 'Sample']

                in_train.drop(dropout, axis=1, inplace=True)
                in_valid.drop(dropout, axis=1, inplace=True)

                xtrain, xtest, ytrain, ytest = train_test_split(in_train, train_target, test_size=3)

                print(xtrain)
                model.fit(xtrain, ytrain)

                same_score = model.score(xtest, ytest)  # Calculating accuracy in the same year

                n_predict = model.predict(in_valid)
                score = model.score(in_valid, valid_target)
                # print(valid_target.shape)
                # print(n_predict.shape)
                obversed = np.squeeze(valid_target)
                print(obversed.shape)
                accuracy = np.corrcoef(n_predict, obversed)[0, 1]
                mse = mean_squared_error(obversed,n_predict)
                print("The accuracy for {} in RM is: {}".format(trait, accuracy))
                print("The mse for {} in RM is: {}".format(trait, mse))
                print("The variance for predicted {} is: ".format(trait,np.var(n_predict)))
                print("A bite of output:")
                print("observe: ", obversed[:50])
                print("predicted: ", n_predict[:50])
                save_df = pd.DataFrame({"obv":obversed,"pred":n_predict})
                save_df.to_csv("~/saved_outcomes.csv",sep="\t")

                avg_acc.append(accuracy)
                avg_score.append(score)
                acg_same_score.append(same_score)
                avg_mse.append(mse)
                r += 1
                records.append([trait, "2013-15", "2017", n_features, same_score, score, accuracy,mse])
            accs.append([trait, "2013-15", "2017", n_features, np.mean(acg_same_score), np.mean(avg_score), np.mean(avg_acc),np.mean(avg_mse)])


    record_train_results(accs,cols=record_cols,method="RM",path="~",para="max_features")
    record_train_results(records,cols=record_cols,method="RM",path="~",para="max_feature_raw")

    """
    results = pd.DataFrame(accs,columns=["trait","trainSet","validSet","n_features","test_score","valid_score","accuracy","mse"])
    print("Result:")
    print(results)
    results.to_csv("~/rm_train_record.csv",sep="\t")
    """

if __name__ == "__main__":
    main()
