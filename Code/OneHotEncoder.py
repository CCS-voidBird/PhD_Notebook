from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv("E:/learning resource/PhD/sugarcane/2017_FibreBlup_2000.csv",sep="\t")
print(data.head(2))
traits = data["FibreBlup"]

data.drop([data.columns[0],"FibreBlup"],inplace=True,axis=1)
data["Region"] = pd.Categorical(data["Region"])
data["Region"] = data.Region.cat.codes
data = data.astype(int)
print(data)
dataset = tf.data.Dataset.from_tensor_slices((data.values, traits.values))
tfdata = tf.constant(data)
tfdata = tf.one_hot(tfdata,4)

#data = np.expand_dims(data, axis=2)
#trans = OneHotEncoder(sparse=False)

#res = trans.fit_transform(data)
#trans.fit(data)

print(tfdata)
print("done")



