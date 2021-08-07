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

def loss_function(distance,deriv=False):  ## use Cross-Entropy function as loss function

    m = distance.shape[1]
    loss = 0
    if deriv == False:
        loss = (1/m)*np.sum(1/2 * np.square(distance))
    return loss

def loss_function2(y_hat,y):

    m = y_hat.shape[1]
    loss = -(1/m)*np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))
    return loss

def loss_function_deriv(distance):

    m = distance.shape[1]
    loss_dev = -(2/m)*np.sum(distance)
    return loss_dev

def loss_function2_deriv(x,y_hat,y):

    m = y_hat.shape[1]
    loss_dev = (1/m) *np.dot(x.T,(np.sum(y_hat-y)))

    return loss_dev

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
        self.weights = np.random.randn(self.cells,input_panels)
        self.bias = np.ones((number_of_cells,1))
        self.activeFunction = ACTIVEFUNCTIONS[act]

    def activate(self,inputData):

        self.inputs = inputData
        self.insideData = np.dot(self.weights,self.inputs) + self.bias
        self.outputs = self.activeFunction(self.insideData)

        return self.outputs

    def get_derivative(self):

        deriv = self.activeFunction(self.insideData,True)

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
        xdim = self._trainDataX.shape[0]
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

    def test_forward(self):
        X = self._testDataX
        for layer in self.layers:

            X = layer.activate(X)

        self.final = X
        return self.final


    def backward_propagation(self):

        y_hat = self.final
        distance = self._trainDataY - y_hat

        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            if layer == self.layers[-1]:
                layer.error = self._trainDataY - y_hat
                print("loss")
                print(loss_function2(y_hat,self._trainDataY))
                layer.delta = np.dot(loss_function2_deriv(layer.insideData,y_hat,self._trainDataY), layer.get_derivative())
                #print("detla:",layer.delta.shape)
                #print("dev: ",loss_function_deriv(distance))
                #print(layer.get_derivative().shape)
                #print(layer.weights.shape)
                layer.weights += self.learnRate * np.dot(layer.delta,self.layers[idx-1].outputs.T)
                layer.bias += self.learnRate * np.dot(layer.delta, np.ones((layer.delta.shape[1], 1)))
                #print("get update weight")


            else:
                next_layer = self.layers[idx+1]
                #print("update next weight")
                #print((next_layer.weights.T *next_layer.delta).shape)
                #print(layer.get_derivative().shape)
                #layer.delta =  (next_layer.weights.T*next_layer.delta).dot(layer.get_derivative())
                layer.delta = np.dot(next_layer.weights.T,next_layer.delta) * layer.get_derivative()
                layer.weights += self.learnRate * np.dot(layer.delta, self.layers[idx - 1].outputs.T)
                layer.bias += self.learnRate * np.dot(layer.delta, np.ones((layer.delta.shape[1],1)))



    def train(self,reps=10):
        mses = []
        for rep in range(reps):
            self.forward()
            self.backward_propagation()
            if loss_function(self._trainDataY-self.final) == 0:
                break
            if rep % 10 == 0:
                mse = np.mean(np.square(self._trainDataY - self.final))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' %(rep, float(mse)))

    def predict(self):
        y_predict = self.test_forward()  # 此时的 y_predict 形状是 [600 * 2]，第二个维度表示两个输出的概率
        y_predict = np.argmax(y_predict, axis=1)
        print("DONE")
        print(np.sum(y_predict == self._testDataY) / len(self._testDataY))
        return y_predict

def main():
    X, y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,)
    print(X.shape, y.shape)  # (1000, 2) (1000,)
    mlp = MLP(X_train.T,y_train.T,X_test.T,y_test.T)
    #mlp.init_layer(3,"Sigmoid")
    mlp.init_layer(3,"Sigmoid")
    mlp.init_layer(1,"Sigmoid")
    mlp.train()
    mlp.predict()
    print(y_test.T)
    for layer in mlp.layers:
        print(layer.weights)


if __name__ == "__main__":
    main()