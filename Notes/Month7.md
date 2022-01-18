Month 6 Notebook
====

Date: 01/Jan/2021 - 31/Jan/2021

<br>
----



Goal:

1. Finish 1st sugarcane ML draft
2. milestone 1 draft
3. Modify ML system to fully config file based.
4. Sort current results
4. [Abstract - AGBT](https://www.agbt.org/events/agbt-ag/abstract/) conference



read list:

+ Breiman, L. Random Forests. *Machine Learning* **45,** 5–32 (2001). https://doi.org/10.1023/A:1010933404324
+ G. Louppe, “Understanding Random Forests: From Theory to Practice”, PhD Thesis, U. of Liege, 2014.

+ Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." *European conference on computer vision*. Springer, Cham, 2014.

Sort ML knowledge:



decision tree ==> random forest==> feature importance evaluation 

multilayer perceptron ==> convolutional neural network

mean normalized discounted cumulative gain value (MNV)

![image-20220114172238179](E:\learning resource\PhD\PHD_Notebook\Notes\MNV.png)

where $d(i)=1/(log_2i+1)$ is a monotonically decreasing discount function at position *i*; *y*(*i*, *Y*) is the *i*th value of observed phenotypic values *Y* sorted in descending order, here *y*(1, *Y*) ≥y(2,Y)≥…y(n,Y);y(i,X)≥y(2,Y)≥…y(n,Y);y(i,X) is the corresponding value of *Y* in the score pairs (*X*, *Y*) for the *i*th value of predicted scores *X* sorted in descending order. Thus, MNV has a range of 0 to 1 when all the observed phenotypic values are larger than zero; a higher MNV(*k, X, Y*) indicates a better performance of the GS model to select the top-ranked *k* (*α* = *k*/2000, 1 ≤ *k* ≤ 2000, 1% ≤ *α* ≤ 100%) individuals with high phenotypic values.

The entire tenfold cross-validation experiment was repeated ten 
