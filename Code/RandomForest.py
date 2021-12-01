from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from Functions import *
from sklearn.datasets import make_regression

def rf_train(train_data,valid_data):

    pass


def main():
    X, y = make_regression(n_features=4, n_informative=2,random_state = 0, shuffle = False)
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(X, y)
    print(regr.predict([[0, 0, 0, 0]]))

if __name__ == "__main__":
    main()
