Fundamental Deep neural network



basic strategy:

+ input layer (0th layer)
+ hidden layer

Activation functions

sigmoid:

$Equation:$	$f(x)=\frac{1}{1+e^{-x}}$

$Derivative:$	$\frac{d(y)}{d(x)}=f(x)(1-f(x))$



![Activation Functions and their Derivatives](https://editor.analyticsvidhya.com/uploads/94131Screenshot%20(43).png)



Multilayer perceptron study; current problems:

+ Backward propagation
+ loss function
+ core theory in math
+ more hyper-parameters
+ run code with numpy module 





**Partial derivative**

+ $f^{'}_x = \frac{\partial{f}}{\part{x}}$
+ $\delta$



Cross-Entropy Function:

![image-20210806213722473](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20210806213722473.png)





CNN

+ sub-sampling: (inputSize - kernelSize)/step + 1
+ padding: add 1(mostly) to edges of previous layers for keep the size of convolutional output 
+ Dropout: randomly/temporally delete hidden layers for avoid over-fitting 

![](E:\learning resource\PhD\PHD_Notebook\Notes\conv_bias_eq.png)


