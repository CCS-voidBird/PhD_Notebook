Month 2 Notebook
====

Date: 02/Aug/2021 - 31/Aug/2021

<br> Editor: Chensong Chen
----

Current Goals:

+ Optimize MLP model;
+ read papers for combining genetic data and DL model;
+ Start generate CNN demo;

Read list:

+ Pooling Methods in Deep Neural Networks, a Review, Hossein Gholamalinezhad1, Hossein Khosravi





1. Multilayer perceptron summary

Training sample array (N x M) and binary labels Y (1 x M)

Input Layer

Hidden Layers ($l_n$)

output Layer (O)

Hidden_activation function (af)

Output activation function (afo)

loss function (Loss)

Weights in each hidden layers ($W_n$ x $W_{n-1}$​​)

Bias in each layers (1 x $B_{n-1}$​​)​



M: number of samples

N: number of features

W: number of weight

B: number of Bias 



**START**

Input Layer get &rarr; samples (N x M) 

Transfer to Hidden layer $I$​​​ with weights ($N_l$,$N_{l-1}):$

​	While weights in this layer is an {i x j} array:

​	 $Z^I_{ij}$​​ = $w_{ij}x_m + b_n$​​



​	The output of this array:

​	$A^I_{ij} = af(Z^I_{ij})$​  And the output should be a ($N_l$ x M) array



The output array:

​	$A^O = af_o(w^O_{ij}A_{O-1} + b^O_n) $​​



Loss function (Cross-Entropy Function):

​	$f_{LOSS}(A^O,Y) = $​ $\sum −(ylog(p)+(1−y)log(1−p))$​   # a float

​	$\frac{\partial{f_{loss}}}{\partial{A^O}} = -\frac{y}{A^O} + \frac{1-y}{1-A^O}$​​  ​​# a float



For Softmax derivative:

​	$\frac{\partial{f_{loss}}}{\partial{A^O}} = A^O(1-A^O)$​ ​



Back forward propagation:



For Weights of output layer:

​	$\delta \frac{\partial{f_{loss}}}{\partial{Z^O}} = $$\frac{\partial{f_{loss}}}{\partial{A^O}}\frac{\partial{afo}}{\partial{Z^O}}$​​ = $\delta$​ # an ($N_y$ x M) array

  $\frac{\partial{afo}}{\partial{Z^O}}$ = ${af}^{'}(Z^O)$  # an ($N_y$ x M) array 

 $w_O'$ = $\frac{\partial{f_{loss}}}{\partial{W^O}}$ = $\delta \frac{\partial{afo}}{\partial{Z^O}} \frac{\partial{Z^O}}{\partial{w^O}}$ = $f'_{loss}(A^O).af'(Z^O).A_{L-1}^T$  # ($N_y$ x M) $\dot{}$ (M x $N_{l-1}$) = ($N_y$ x $N_{l-1}$) fit the size of w



2. Start CNN (convolutional neural network) learning 



Basic strategy:

+  Input Layer
+ Convolutional layer
  + 
+ Pooling Layer
  + Layer types:
    + Max-Pooling
      + for each sub pool $M_i$(n x n), output a avg matrix $(\frac{M}{n} \times \frac{M}{n})$, the value should be $\max(M_i)$​
    + AVG-Pooling
      + for each sub pool $M_i$(n x n), output a avg matrix $(\frac{M}{n} \times \frac{M}{n})$, the value should be $\frac{\sum M_i}{n^2}$
    + Mixed-Pooling
      + A mixture of Max-Pooling and AVG-pooling, $\lambda$​ is a modified parameter
      + $s_j = \lambda \max(M_i) + (1-\lambda)AVG(M_i) $​
    + $L_p$​ Pooling
      + $s_j = (\frac{1}{\abs{M_i}}\sum M_j)^{\frac{1}{p}}$
      + $p$​ is a modified parameter, while p =1, the pooling performs as the AVG pooling; while $p = \infty$, it performs as the MAX pooling. 
    + Stochastic Pooling
      + A pooling method based on activation weight and random selection
      + $p_i$ = $\frac{a_i}{\sum{W_i}}$
      + $Selection = random(M_{pi}) $
    + Spatial Pyramid Pooling
      + A stack layer which resotres each level of convolutional layers (from basic to above)
    + Region of Interest Pooling
  + Layer selection methods:
    + Multi-scale order-less Pooling
    + Super-pixel Pooling
    + PCA network
    + Compact Bilinear Pooling
+ fully connected layer
+ Receptive layer  

