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
    #print(y)
    #print("y")
    m = y.shape[0]
    loss = 0
    if deriv == False:
        loss = -(1/m)*np.sum(np.abs(y-yt))
    elif deriv == True:
        loss = -(2/m)*(np.sum(np.abs(y-yt)))
    return loss

def weight_update(number_of_layers, layerUnits, y, layer_output,weights,bias):
    total_err = y-layer_output[-1]
    print(weights)
    for level in range(number_of_layers-1):
        layer = -level-1
        print("current weight")
        print(weights[-level-1] )
        if level == 0:
            err_del = total_err * sigmoid(layer_output[layer],True)
            print("shape is ",err_del.shape)
        else:
            err_del = np.dot(weights[layer+1].T,err_del)*sigmoid(layer_output[level])
        #print("ols: ",weights[layer])
        #print("new: ",layer_weight_update(err_del,layer_output[layer],weights[layer]))
        weights[layer] += layer_weight_update(err_del,layer_output[layer-1])
        #bias[layer] += layer_bias_update(err_del)
        print()
    return weights,bias

def layer_weight_update(err_del,z_prev, learnRate=0.1):

    e_d = err_del
    z_r = z_prev.T
    return np.dot(e_d,z_r) * learnRate

def layer_bias_update(err_del,learnRate=0.1):

    e_d = err_del
    w_d = learnRate * e_d

    return w_d

def switch(x):
    x[x>=0.5] = 1
    x[x<0.5] = 0
    return x


def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(1, -1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predic = model.predict(X_new)
    zz = y_predic.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF590', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

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
        self.number_of_units = None

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
        number_of_outputUnit = 2 #Y.ndim
        self.number_of_units = [X.ndim] + [layerUnits for unit in range(hidden_layers)] + [number_of_outputUnit]
        self.weights = []  # np.random.randn(layerUnits,number_of_inputUnit)
        self.bias = [] ## np.random.randn(1,layerUnits)
        for idx in range(len(self.number_of_units)-1):
            col = self.number_of_units[idx]
            row = self.number_of_units[idx+1]
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
        self.layers = [0 for x in range(len(self.weights)+1)]
        print("Total layers:",len(self.layers))
        print("Total bias: ",len(self.bias))

    def training(self,redo = 300):

        l0,y = self.input
        self.layers[0] = l0
        print(len(self.weights))
        #print(self.weights)
        for rep in range(redo):
            #print(self.weights)
            #print(self.bias)
            #self.hiddenLayers[0] = self.activeFunction(self.weights[0].dot(l0), False)
            print("start forward")
            for layer in range(1,len(self.layers)): ## The input Layer is the 1st layer;
                self.layers[layer] = self.activeFunction(self.bias[layer-1] + self.weights[layer-1].dot(self.layers[layer-1]),False)
                print(layer)
            #print(self.layers[-2])
            self.outputLayer = self.layers[-1]
            #self.layers[-1] = self.outputLayer

            self.weights,self.bias = weight_update(len(self.layers),self.number_of_units,y,self.layers,self.weights,self.bias)
            """
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
            """
            if rep ==0:
                print(self.weights)
                print(self.bias)
            if rep % 100 == 0 or rep == redo-1:
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