import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

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




class Layer:

    def __init__(self,number_of_cells,input_panels,act):

        self.cells = None
        self.weights = None
        self.bias = None
        self.inputs = None
        self.insideData = None
        self.outputs = None
        self.activeFunction = None
        self.error = None
        self.delta = None
        self.label = None
        self.initialization(number_of_cells,input_panels,act)

    def initialization(self,number_of_cells,input_panels,act):

        self.cells  = number_of_cells
        print(input_panels)
        self.weights = np.random.randn(input_panels,self.cells)
        self.bias = np.random.randn(number_of_cells)
        self.activeFunction = ACTIVEFUNCTIONS[act]

    def activate(self,inputData):

        self.inputs = inputData
        self.insideData = np.dot(self.inputs,self.weights) + self.bias
        self.outputs = self.activeFunction(self.insideData)

        return self.outputs

    def get_derivative(self):

        deriv = self.activeFunction(self.outputs,True)

        return deriv

    def back_update(self,error):

        pass

class MLP:

    def __init__(self,trainX,trainY,testDataX,testDataY):

        #self._data = None

        self._trainDataX = trainX
        self._trainDataY = trainY
        self._testDataX = testDataX
        self._testDataY = testDataY
        self.layers = []
        self.layerUnits = []
        self.learnRate = 0.1
        self.final = None

    def init_layer(self,number_of_units,act):
        xdim = self._trainDataX.shape[1]
        if len(self.layers) == 0:
            layer = Layer(number_of_units,xdim,act)
        else:
            inputUnits = self.layers[-1].cells
            layer = Layer(number_of_units, inputUnits, act)
        self.layers.append(layer)
        self.layerUnits.append(number_of_units)

    def initialization(self,trainData,testData):

        self._trainData = trainData
        self._testData = testData

    def forward(self):
        X = self._trainDataX
        for layer in self.layers:

            X = layer.activate(X)

        self.final = X

    def backward_propagation(self):

        y_hat = self.final

        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            if layer == self.layers[-1]:
                layer.error = self._trainDataY - y_hat
                layer.delta = layer.error * layer.get_derivative()
                print(layer.get_derivative().shape)
                print(layer.error.shape)
            else:
                layer.error = np.dot(self.layers[idx+1].weights, self.layers[idx+1].delta)
                layer.delta = layer.error * layer.get_derivative()

        for i in range(len(self.layers)):
            layer = self.layers[i]

            if layer == self.layers[0]:
                layer.weights += layer.delta * self.learnRate * self._trainDataX
            else:
                layer.weights += layer.delta * self.layers[i-1].outputs.T * self.learnRate

    def train(self,reps=100):
        mses = []
        for rep in range(reps):
            self.forward()
            self.backward_propagation()
            if rep % 10 == 0:
                mse = np.mean(np.square(self._trainDataY - self.final))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' %(rep, float(mse)))


def main():
    X, y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X.shape, y.shape)  # (1000, 2) (1000,)
    mlp = MLP(X_train,y_train,X_test,y_test)
    mlp.init_layer(3,"Sigmoid")
    mlp.init_layer(3,"Sigmoid")
    mlp.init_layer(1,"Sigmoid")
    mlp.train()



if __name__ == "__main__":
    main()