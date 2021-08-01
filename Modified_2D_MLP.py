import pandas as pd
import numpy as np

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
        self.outputWeight = None
        self.hiddenLayers = None
        self.outputLayer = None
        self.outputFunction = None
        self.learnRate = 1

    def readData(self,data):

        data.replace("A", 1, inplace=True)
        data.replace("B", 0, inplace=True)
        self.data = data
        self.train_data = data[0:len(data) // 2][:30]
        self.test_data = data[len(data) // 2:-1][:30]

    def initialization(self,hidden_layers = 4, layerUnits = 2,active = "ReLu"):

        X = np.array([self.train_data["x"].values.tolist(), self.train_data["y"].values.tolist()])
        Y = np.array(self.train_data["label"].values[:30])

        number_of_inputUnit = X.ndim
        number_of_outputUnit = Y.ndim
        self.weights = [np.random.randn(number_of_inputUnit,layerUnits)]
        for layer in range(hidden_layers):
            self.weights.append(np.random.randn(layerUnits,layerUnits))
        self.outputWeight = np.random.randn(layerUnits,number_of_outputUnit)
        self.activeFunction = ACTIVEFUNCTIONS[active]
        self.outputFunction = sigmoid
        self.input = [X,Y]
        self.hiddenLayers = [0 for x in range(len(self.weights))]

    def training(self,redo = 5):

        l0,y = self.input
        print(self.outputWeight)

        #print(self.weights)
        for rep in range(redo):

            self.hiddenLayers[0] = self.activeFunction(self.weights[0].dot(l0), False)
            for layer in range(1,len(self.weights)):
                self.hiddenLayers[layer] = self.activeFunction(self.weights[layer].dot(self.hiddenLayers[layer-1]),False)

            self.outputLayer = self.outputFunction(self.outputWeight.T.dot(self.hiddenLayers[-1]),False)
            l_error = y - self.outputLayer
            if rep == 0:
                print("while 0")
                print(self.hiddenLayers[-1])
                print("out:",self.outputLayer)
            #print(l_error.shape)
            l_delta = l_error*self.outputFunction(self.outputLayer, True) ## considering np.dot
            #print("update")
            #print(np.dot(l_delta,self.hiddenLayers[-1].T).reshape(2,1))
            #print(self.learnRate*np.dot(l_delta,self.hiddenLayers[-1].T).reshape(2,1))
            #print(self.outputWeight)
            #print(self.outputWeight + self.learnRate*np.dot(l_delta,self.hiddenLayers[-1].T).T)
            self.outputWeight += self.learnRate*np.dot(l_delta,self.hiddenLayers[-1].T).T


            for back_lay in range(len(self.weights)):
                l_delta = l_error * self.activeFunction(self.hiddenLayers[back_lay-1], True)
                if rep == 0:
                    print("delta:")
                    print(l_delta)
                    print(np.dot(l_delta,self.hiddenLayers[back_lay-1].T).T)
                #print(self.weights[back_lay-1])
                self.weights[back_lay-1] += self.learnRate * np.dot(l_delta,self.hiddenLayers[back_lay-2].T).T

            if rep % 100 == 0 or rep == redo-1:
                print(np.mean(np.abs(l_error)))
                print("updata:")
                print(self.weights)
                #print(self.weights)
        #print(self.weights)
        #print(self.outputWeight)
        print("DONE")
        print("input:",l0)
        print("output:",self.outputLayer)
        print("excepted:",y)
        return self.outputLayer

def main():

    data = pd.read_csv("./pts.csv",sep="\t")
    mlp = MLP()
    mlp.readData(data)
    mlp.initialization(active="ReLu")
    mlp.training()


if __name__ == "__main__":
    main()