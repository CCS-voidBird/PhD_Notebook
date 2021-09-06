Month 3 Notebook
====

Date: 01/Sept/2021 - 31/Sept/2021

<br> Editor: Chensong Chen
----



Current Goals:

+ Write some draft (introduction, theory work) [in working]

+ Setting up a validation method for performance comparison [in working]

+ Optimize AlexNet model [week 2]

+ Add interaction features to AlexNet [tending] 

+ Test performance per EnvSet (Series, Region, Crop) 

+ Try to build ResNet (Residual Network)

+ Create separated train dataset and test dataset (For both GBLUP and DL) by region and years.




Read list:

+ Pisaroglo De Carvalho M, Gezan SA, Peternelli LA, Pereira Barbosa MH (2014) Estimation of additive and nonadditive genetic components of sugarcane families using multitrait analysis. Agron J 106:800–808. https://doi.org/10.2134/agronj2013.0247
+ Castro, W.; Marcato Junior, J.; Polidoro, C.; Osco, L.P.; Gonçalves, W.; Rodrigues, L.; Santos, M.; Jank, L.; Barrios, S.; Valle, C.; Simeão, R.; Carromeu, C.; Silveira, E.; Jorge, L.A.d.C.; Matsubara, E. Deep Learning Applied to Phenotyping of Biomass in Forages with UAV-Based RGB Imagery. *Sensors* **2020**, *20*, 4802. https://doi.org/10.3390/s20174802
+ Meuwissen THE, Hayes BJ, Goddard ME (2001) Prediction of total genetic value using genome wide dense marker maps. Genetics 157:1819–1829
+ Bijma P, Wientjes YCJ, Calus MPL. Breeding Top Genotypes and Accelerating Response to Recurrent Selection by Selecting Parents with Greater Gametic Variance. *Genetics*. 2020;214(1):91-107. doi:10.1534/genetics.119.302643

+ Eraslan, G., Avsec, Ž., Gagneur, J. *et al.* Deep learning: new computational modelling techniques for genomics. *Nat Rev Genet* **20,** 389–403 (2019). https://doi.org/10.1038/s41576-019-0122-6



Deep Learning Draft

1. Machine Learning concept
   + Supervised model
   + Unsupervised model
2. Neural network
   + A typical neural network is an organic integrity of artificial neurons, connection layers and methmatic/transmission functions for simulating biological brains and achieving predictions. As the basic unit of neural network, an artificial neurons contains a input portal, a weight $w$ for measuring neuron input, a bias $b$, an inner-procession algorithm $z$ (e.g. $z = wx +b$), a tempory memory for restore processed signal and an output portal. 
   + **[Add a pragraph to explain neurons]** The role of neurons is process the data by its build-in function and weight, a single neuron probably won't have exact meaning in reality.
   + A certain number $N$ of artificial neuron (as far as channels) can contribute into a layer (can be defined as hidden layer or output layer), each layer contain a weight&bias matrix ($N_{output} \times N_{input}$). By default, the input data will be reformatted as a ($N_ {input}\times M$) matrix; the input layer won't perform any calculations (as long as transformation) to the data, then the first hidden layer receives the data, it feed the data into a activate function $Af$ such as sigmoid, softmax and linear, the process can be described as $A = Af(wx +b)$, and the dimension of the output will be a ($N_{output} \times M$) matrix until the output layer which has a $(N \times N_y)$ weight matrix exports the predicted $\hat{y}$. The above process is also be named as feed-forward propagation.
   + As in the real cases with big data, the predict of the feed-forward propagation of a typical neural network (such as MLP), will be a $(1\times M)$ vector (for single label) or $(N \times M)$ matrix for $N$ labels, there will be a loss function ($L$) such as MSE (Mean Squared Error) and  Cross Entropy to measure distances between prediction and observation. The overall error will be $Error = L(\hat{y},y)$
   + After getting total error, there are several options for update weight/bias matrix. The [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) method has been a widely used approach for reduce error rate in many kinds of models. 
3. Deep learning
   + Deep learning methodology is a sub-type of artificial neural network which is also classified into machine learning algorithm [[1]](#1). The main difference between deep learning and conventional neural network is the neural network would basically transmit data though layers, while the deep learning models tend to associate signal selection and transformation.
4. CNN
5. MLP/CNN in genetic research
6. 



> There are two main benefits to taking only a small random subset of the training set at each optimization step rather than the full training set. First, the algorithm requires a constant amount of memory regardless of the data set size, which allows models to be trained on data sets much larger than the available memory. Second, the random fluctuations between batches were demonstrated to improve the model performance by [regularization](https://www.nature.com/articles/s41576-019-0122-6#Glos45) [[2]](#2)



Reference 

<a name="1">[1]</a> Deng, Li, and Dong Yu. "Deep learning: methods and applications." *Foundations and trends in signal processing* 7.3–4 (2014): 197-387.

<a name="2">[2]</a> Eraslan, G., Avsec, Ž., Gagneur, J. *et al.* Deep learning: new computational modelling techniques for genomics. *Nat Rev Genet* **20,** 389–403 (2019). https://doi.org/10.1038/s41576-019-0122-6