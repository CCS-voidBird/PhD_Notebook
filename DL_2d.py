import pandas as pd
import numpy as np

data = pd.read_csv("./pts.csv",sep="\t")
data[data["label"] == "A"]["label"] = int(1)
data[data["label"] == "B"]["label"] = int(0)
pts = data[["x","y"]].values
data.replace("A",1,inplace=True)
data.replace("B",2,inplace=True)
test = data[0:len(data)//2]
train = data[len(data)//2:-1][:10]
X = np.array([train["x"].values.tolist(),train["y"].values.tolist()])
Y = np.array(train["label"].values[:10])
print(X.shape)
print(X)
print(Y)
#print(Y.shape)

def reLu(x,deriv=False):
    if deriv == False:
        #print(np.maximum(x,0))
        return np.maximum(x,0)
    else:
        return 1*(x>0)

def sigmoid(x,deriv=False):
    s = 1/(1+np.exp(-x))
    ds = x*(1-x)
    if deriv==True:
        return ds
    return s

def cost_function(x):
    pass

dims = [2 for num in range(10)]

np.random.seed(1)
ws=[np.random.randn(2,2)]
for r in range(3):
    ws.append(np.random.randn(2,2))
ws_out = np.random.randn(2,1)
print(ws[1].shape)

lns = [x for x in range(4)]
reps = 1000
for rep in range(reps):
    l0 = X
    print(rep,"times train")
    print(l0.shape)
    for lay in range(len(ws)):
        print(lay)
        if lay ==0:
            print(ws[0])
            #print(np.dot(ws[0].T,l0))
            #lns[lay] = sigmoid(np.dot(ws[0].T,l0),False)
            lns[lay] = sigmoid(ws[0].T.dot(l0), False)

        else:
            lns[lay] = sigmoid(ws[lay].T.dot(lns[lay-1]),False)
            #lns[lay] = sigmoid(np.dot(ws[lay].T, lns[lay - 1]), False)

    print(lns)
    l_final = sigmoid(ws_out.T.dot(lns[-1]),False)
    lns[-1] = l_final
    #print(l_final)
    l_error = Y - l_final
    if rep % 100 == 0:
        print("current:")
        print(np.mean(np.abs(l_error)))
        print(l_error)
    l_delta = l_error * sigmoid(ws_out.dot(lns[-1]),True)
    print(l_delta)


    for lay in range(len(lns)):
        print("reverse")
        if lay == 0:
            perr = Y - l_final
            pdel = perr * sigmoid(ws[-1].dot(lns[-2]),True)
            print(pdel)
            ws_out += l_delta.dot(lns[-2].T)
        else:
            pdl = pdel * sigmoid(ws[-lay-1].dot(lns[-lay-1]))
            ws[-lay-1] += pdl.dot(lns[-lay-2].T)


#print(sigmoid(X,deriv=False))