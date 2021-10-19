from sklearn.preprocessing import OneHotEncoder
import pandas as pd

data = pd.read_csv("E:/learning resource/PhD/sugarcane/2017_FibreBlup_2000.csv",sep="\t")
print(data.head(2))
traits = data["FibreBlup"]

data.drop("FibreBlup",inplace=True,axis=1)
trans = OneHotEncoder()

trans.fit(data)

print(trans.categories_)



