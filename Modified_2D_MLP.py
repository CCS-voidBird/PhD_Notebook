import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



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

def taNh(x,deriv=False):
    if deriv == False:
        return np.tanh(x)
    else:
        return 1-x**2

def single_backward_propagation(dA_curr,w_curr,b_curr,z_curr,a_prev,active = "Sigmoid"):

    activeF = ACTIVEFUNCTIONS[active]



def sigMoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def loss_function(y,yt,deriv=False):
    print(y.shape)
    print("y")
    m = y.shape[1]
    loss = 0
    if deriv == False:
        loss = -(1/m)*np.sum(np.abs(y-yt))
    elif deriv == True:
        loss = -(2/m)*(np.sum(np.abs(y-yt)))
    return loss

def weight_update(err,z_curr,z_prev,activef = "sigmoid", learnRate=0.1):

    af = ACTIVEFUNCTIONS[activef]
    e_d = loss_function(err,deriv=True)
    a_d = af(z_curr,deriv=True)
    z_d = z_prev

    w_d = learnRate * a_d * e_d * z_d
    return w_d

def switch(x):
    x[x>=0.5] = 1
    x[x<0.5] = 0
    return x

ACTIVEFUNCTIONS={
    "Sigmoid": sigmoid,
    "ReLu":reLu,
    "tanh":taNh
}



class MLP:

    def __init__(self):
        self.data=None
        self.train_data = None
        self.test_data = None
        self.activeFunction=None
        self.weights = None
        self.input = None
        self.output = None
        self.layers = None
        self.outputWeight = None
        self.hiddenLayers = None
        self.outputLayer = None
        self.outputFunction = None
        self.learnRate = 0.1
        self.bias=None
        self.outBias = None

    def readData(self,data):

        data.replace("A", 1, inplace=True)
        data.replace("B", 0, inplace=True)
        self.data = data
        self.train_data = data[0:len(data) // 2][:50]
        self.test_data = data[len(data) // 2:-1][:50]

    def initialization(self,hidden_layers = 1, layerUnits = 3,active = "ReLu"):

        X = np.array([self.train_data["x"].values.tolist(), self.train_data["y"].values.tolist()])
        Y = np.array(self.train_data["label"].values[:50])


        plt.scatter(x=X[0],y=X[1])
        plt.show()

        number_of_inputUnit = X.ndim
        number_of_outputUnit = Y.ndim
        number_of_layerUnit = [X.ndim] + [layerUnits for unit in range(hidden_layers)] + [Y.ndim]
        self.weights = []  # np.random.randn(layerUnits,number_of_inputUnit)
        self.bias = [] ## np.random.randn(1,layerUnits)
        for idx in range(len(number_of_layerUnit)-1):
            col = number_of_layerUnit[idx]
            row = number_of_layerUnit[idx+1]
            self.weights.append(np.random.randn(row,col))
            self.bias.append(np.random.randn(row,1))
        print("length: ",len(self.weights),len(self.bias))
        #self.outputWeight = np.random.randn(number_of_outputUnit,layerUnits)
        #self.weights.append(np.random.randn(number_of_outputUnit,layerUnits))
        #self.bias.append(np.random.randn(1,Y.ndim))
        self.activeFunction = ACTIVEFUNCTIONS[active]
        self.outputFunction = sigmoid
        #self.outBias = np.random.randn(1)
        self.input = [X,Y]
        self.hiddenLayers = [0 for x in range(len(self.weights)+1)]
        self.layers = [0 for x in range(len(self.weights)+2)]
        print("Total layers:",len(self.layers))
        print("Total bias: ",len(self.bias))

    def training(self,redo = 1):

        l0,y = self.input
        self.layers[0] = l0
        print(len(self.weights))
        #print(self.weights)
        for rep in range(redo):
            print(self.weights)
            print(self.bias)
            #self.hiddenLayers[0] = self.activeFunction(self.weights[0].dot(l0), False)
            print("start forward")
            for layer in range(1,len(self.layers)-1): ## The input Layer is the 1st layer;
                self.layers[layer] = self.activeFunction(self.bias[layer-1].T + self.weights[layer-1].dot(self.layers[layer-1]),False)
                print(layer)
            print(self.layers[-2])
            self.outputLayer = self.outputFunction(self.bias[-1], + self.weights[-1].dot(self.layers[-2]),False)
            self.layers[-1] = self.outputLayer
            print("SDFGHJKL")
            print(self.outputLayer)
            l_error = loss_function(y,self.outputLayer)
            print(l_error)
            l_delta = l_error*self.outputFunction(self.outputLayer, True) ## considering np.dot
            print(l_delta)


            ### bachward_propagate processing
            print(len(self.weights))
            for back_lay in range(len(self.weights)):
                print("reverse: {}".format(back_lay))
                reverse = -1-back_lay
                if reverse == -1:
                    l_delta = l_error*self.outputFunction(self.outputLayer, True)
                    self.weights[-1] -= weight_update(l_error,self.outputLayer,self.hiddenLayers[-1])

                else:
                    print("1213231231231")
                    print(l_delta.shape)
                    print((self.weights[reverse+1].T*l_delta).shape)
                    print(self.activeFunction(self.hiddenLayers[reverse], True).shape)
                    l_error = self.activeFunction(l_error)
                    l_delta = self.activeFunction(self.hiddenLayers[reverse], True)*(self.weights[reverse+1].T*l_delta)

                    # l_delta * self.weights[reverse+1].dot(self.activeFunction(self.hiddenLayers[reverse], True))
                    #l_delta = l_delta * self.activeFunction(self.hiddenLayers[reverse], True).dot(self.weights[reverse+1])
                if rep == 0:
                    print("delta11:")
                    print(l_delta)
                try:
                    self.weights[reverse] -= self.learnRate * (np.dot(l_delta,self.hiddenLayers[reverse-1].T)+self.weights[reverse])
                except:
                    print("go")
                    print((np.dot(l_delta,self.hiddenLayers[reverse-1].T).T+self.weights[reverse]).shape)
                #print("############")
                #print()
                self.bias[reverse] -= self.learnRate * (np.mean(l_delta))
            if rep ==0:
                print(self.weights)
                print(self.bias)
            if rep % 100 == 0 or rep == redo-1:
                print(np.mean(np.abs(l_error)))
                print("updata:")
                print(self.weights)
                #print(self.weights)
        #print(self.weights)
        #print(self.outputWeight)
        print("DONE")
        #print("input:",l0)
        print(self.bias)
        print("output:",self.outputLayer)
        print("as: ", switch(self.outputLayer))
        print("excepted:",y)
        return self.outputLayer

def main():

    data = pd.read_csv("./pts.csv",sep="\t")
    mlp = MLP()
    mlp.readData(data)
    mlp.initialization(active="Sigmoid")
    mlp.training()


if __name__ == "__main__":
    main()