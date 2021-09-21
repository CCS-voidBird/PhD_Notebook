Day1

lecture0

​	AI >> ML >> DL

​	Electronic brain -> Perceptron -> ADALINE -> XOR -> Multi-layer perceptron -> SVM -> Deep Neural network

Innovations:

​	CPUs, GPUs, TPUs(Tensor Processing Units: designed specifically for deep learning)

Innovations in Algorithm:

​	backpropagation/gradient propagation

​	better activate functions

​	better optimize ways



The State of the art

clockwork universe  -> determinism 

Isaac Newton





Day2



Kares workflow

+ Prepare and split data
  + Numpy arrays
  + from keras.preprocessing.image import ImageDataGenerator
+ Define the model
  + from keras.layers import Dense, Dropout, activation, Flatten...)
+ Compile the model (model.compile)
+ fit the model
+ predict result for unknow value (model.evaluate())
+ Modify until satisfied
+ Save for future use



Logistic regression

​	logit($\phi(z) $) = $\beta_0 + \beta_1 x$ 

optimiser

learning rate

local minimum -> momentum