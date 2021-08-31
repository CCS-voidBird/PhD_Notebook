Month 2 Notebook
====

Date: 02/Aug/2021 - 31/Aug/2021

<br> Editor: Chensong Chen
----

Current Goals:

+ Optimize MLP model;
+ read papers for combining genetic data and DL model;
+ Start generate CNN demo;
+ sugarcane concept

Add to Read list:

+ [DONE] Pooling Methods in Deep Neural Networks, a Review, Hossein Gholamalinezhad1, Hossein Khosravi
+ [DONE] Kemper KE, Bowman PJ, Pryce JE, Hayes BJ, Goddard ME. Long-term selection strategies for complex traits using high-density genetic markers. J Dairy Sci. 2012 Aug;95(8):4646-56. doi: 10.3168/jds.2011-5289. 
+ [DONE]Meuwissen TH. Maximizing the response of selection with a predefined rate of inbreeding. J Anim Sci. 1997 Apr;75(4):934-40. doi: 10.2527/1997.754934x. PMID: 9110204.
+ [Working]Goddard, M. Genomic selection: prediction of accuracy and maximisation of long term response. *Genetica* **136,** 245–257 (2009). https://doi.org/10.1007/s10709-008-9308-0
+ [DONE]Yadav, S., Wei, X., Joyce, P. *et al.* Improved genomic prediction of clonal performance in sugarcane by exploiting non-additive genetic effects. *Theor Appl Genet* **134,** 2235–2252 (2021). https://doi.org/10.1007/s00122-021-03822-1
+ Pisaroglo De Carvalho M, Gezan SA, Peternelli LA, Pereira Barbosa MH (2014) Estimation of additive and nonadditive genetic components of sugarcane families using multitrait analysis. Agron J 106:800–808. https://doi.org/10.2134/agronj2013.0247
+ Castro, W.; Marcato Junior, J.; Polidoro, C.; Osco, L.P.; Gonçalves, W.; Rodrigues, L.; Santos, M.; Jank, L.; Barrios, S.; Valle, C.; Simeão, R.; Carromeu, C.; Silveira, E.; Jorge, L.A.d.C.; Matsubara, E. Deep Learning Applied to Phenotyping of Biomass in Forages with UAV-Based RGB Imagery. *Sensors* **2020**, *20*, 4802. https://doi.org/10.3390/s20174802
+ Meuwissen THE, Hayes BJ, Goddard ME (2001) Prediction of total genetic value using genome wide dense marker maps. Genetics 157:1819–1829
+ Bijma P, Wientjes YCJ, Calus MPL. Breeding Top Genotypes and Accelerating Response to Recurrent Selection by Selecting Parents with Greater Gametic Variance. *Genetics*. 2020;214(1):91-107. doi:10.1534/genetics.119.302643



1. ***Multilayer perceptron summary***

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



**Forward Propagation **

Input Layer get &rarr; samples (N x M) 

Transfer to Hidden layer $I$​​​ with weights ($N_l$,$N_{l-1}):$

​	While weights in this layer is an {i x j} array:

​	 $Z^I_{ij}$​​ = $w_{ij}x_m + b_n$​​



​	The output of this array:

​	$A^I_{ij} = af(Z^I_{ij})$​  And the output should be a ($N_l$ x M) array



The output array:

​	$A^O = af_o(w^O_{ij}A_{O-1} + b^O_n) $​​



Loss function (Cross-Entropy Function):



​	$f_{LOSS}(A^O,Y) = $​​ $\sum −(ylog(p)+(1−y)log(1−p))$​​   # a float



​	$\frac{\partial{f_{loss}}}{\partial{A^O}} = -\frac{y}{A^O} + \frac{1-y}{1-A^O}$​​  ​​# a float



For Softmax derivative:

​	$\frac{\partial{f_{loss}}}{\partial{A^O}} = A^O(1-A^O)$​ ​



**Backward Propagation **



+ For Weights of output layer:

​	$\delta \frac{\partial{f_{loss}}}{\partial{Z^O}} = $$\frac{\partial{f_{loss}}}{\partial{A^O}}\frac{\partial{afo}}{\partial{Z^O}}$​​ = $\delta$​ # an ($N_y$​ x M) array



  $\frac{\partial{afo}}{\partial{Z^O}}$ = ${af}^{'}(Z^O)$  # an ($N_y$​ x M) array 



 $w_O'$ = $\frac{\partial{f_{loss}}}{\partial{W^O}}$ = $\delta \frac{\partial{afo}}{\partial{Z^O}} \frac{\partial{Z^O}}{\partial{w^O}}$ = $f'_{loss}(A^O).af'(Z^O).A_{L-1}^T$  # ($N_y$ x M) $\dot{}$ (M x $N_{l-1}$) = ($N_y$ x $N_{l-1}$) fit the size of w



+ For weights of hidden Layer:

$\delta \frac{\partial f_{loss}}{\partial{z_l}} = \frac{\partial{f_{loss}}}{\partial{z_{l+1}}} \frac{\partial{z_{l+1}}}{\partial{z_l}}$​​ 

$\frac{\partial{z_{l+1}}}{\partial{z_l}} = \frac{\partial{z_{l+1}}}{\partial{a_l}} \frac{\partial{a_{l}}}{\partial{z_l}} = w^{T}_{l+1} \cdot af'(z_l) $ 

$\frac{\partial f_{loss}}{\partial{z_l}} = \frac{\partial{f_{loss}}} {\partial{z_{l+1}}}\frac{\partial{z_{l+1}}}{\partial{a_l}} \frac{\partial{a_{l}}}{\partial{z_l}} = w^{T}_{l+1} \cdot \delta \cdot  af'(z_l)$​​​​ # an ($N_{l} \times M$ ) array

$w'_l = \frac{ \partial f_{loss}}{\partial{z_l}} \frac{\partial z_l}{\partial w_{l}} = w^{T}_{l+1} \cdot \delta _{l+1} \cdot af'(z_l) \cdot A_{l-1}^T $​​  # ($N_{l} \times M$​​ ) $\cdot $​​ ( $M \times N_{l-1}$​​​ ) = $(N_l \times N_{l-1})$​ array



+ For Linear models: (Use MSE loss function)
  + loss = $\frac{1}{2}||y-a^L||^2$
  + $\delta ^L = \frac{\partial loss}{\partial z^L} = (y-a^L)*af'(z^L)$​​​ 
  + $w'_L = \frac{\partial loss}{\partial z^L} = (y-a^L)*af'(z^L) \cdot A_{L-1}$​
  + $w'_l = w^{T}_{l+1} \cdot \delta _{l+1} \cdot  af'(z_l) \cdot A^T_{l-1}$​​   





2. ***CNN (convolutional neural network) learning***



Basic strategy:

+  Input Layer
+ Convolutional layer
  + 
+  Pooling Layer
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
+  fully connected layer
+  Receptive layer [receptive field]
   + focus on particular partial field from previous nerous.



**HINT** Weight regularization 

+ L1: Sum of the absolute weights.
+ L2: Sum of the squared weights.
+ L1L2: Sum of the absolute and the squared weights.



**example hyperparameter** [[6]](#6) (for blue berry and strawberry)

>  This rule evolves a list of CNN models for each phenotypic trait. We optimized the following hyperparameters (values considered within parentheses): activation function (relu, tanh, linear), number of filters (16, 32, 64, 128), regularization (i.e., weight decay in DL terminology, 0, 0.1, 0.01, 0.001), learning rate (0.1, 0.01, 0.001, 0.0025), number of neurons in fully connected layer (4, 8, 12, 16), number of hidden layers (1,5,10), and dropout (0, 0.01, 0.1, 0.2).



AlexNet example: https://zhuanlan.zhihu.com/p/29786939

https://github.com/sloth2012/AlexNet



Genome breeding related:

+ Barriers in animal breeding:

  + Complex genetic architecture -- Thinking: CNN 2D(3D?) prediction

    + highly polygenic phenotypes
    + high requirements for markers

  + Low breeding speed (reproductive and recombination rates)

  + Limitations of high inbreeding necessity 

  + > However, in dairy cattle, this level of co-ancestry is probably undesirably high.

+ Selection algorithms used in the breeding programs.

  + Genotype-building strategy aiming to minimize the distance to the target genotype (ARCSIN)

![steps.png](https://github.com/CCS-voidBird/hivebox/blob/master/steps.png?raw=true)



+ GEBV note

  > In this case the expected effect of a marker is a non-linear function of the data such that apparently small effects are regressed back almost to zero and consequently these markers can be deleted from the model. The accuracy in this case is considerably higher than when marker effects are normally distributed [[7]](#7). 



3. Sugarcane genome overview [Saccharum spp. hybrids]

+ High heterozygous - interspecific origin & high polyploidy [[1]](#1) [[2]](#2); 

+ polyploid nuclear genome and organellar genomes [[5]](#5)

+ ![img](https://www.frontiersin.org/files/Articles/357194/fpls-09-00616-HTML-r2/image_m/fpls-09-00616-g001.jpg)

  [source] Thirugnanasambandam PP, Hoang NV, Henry RJ. The Challenge of Analyzing the Sugarcane Genome. Front Plant Sci. 2018 May 14;9:616. doi: 10.3389/fpls.2018.00616. PMID: 29868072; PMCID: PMC5961476.

+ sub-genome $I$: *S. spontaneum*

  + wild male thin-stalked, low-sugar
  + 8 basic chromosome
  + basic genome size - 750-843MB
  + multiple ploidy level from 40~128 chromosome

+ sub-genome $II$: *S. officinarum* 

  + a female thick-stalked, high-sugar
  + 8x genome; octoploid species
  + basic chromosome - 10; 
  + basic monoploid genome size - ~1GB
  + overall genome size - ~7.88GB

+ heterozygous sugarcane [[5]](#5)

  + 10 basic chromosome
  + mixture of aneuploid and homologous chromosomes

+ Main qualitative traits
  + cane per hectare (TCH)
  + commercial cane sugar (CCS)
  + Fibre content

Environment: for windows 10

​	Python 3.7

​	Pytorch 1.9.0

​	cuda 10.0

​	cudnn 10.2



Framework: 

+ AlexNet (DONE model) 
+ ResNet （Deep residual network）
  + https://zhuanlan.zhihu.com/p/31852747 （Chinese version)
+ VGGNet



<<<<<<< Updated upstream
**AlexNet Model**

data - Sugarcane

+  genetic array - AA =2, AT = 1, TT=0, --(UNKNOW) set 0.01 as pseudo
+ Series, Region, Trial, Crop information - Temporally excluded
+ 2000 samples randomly selected from origin file
+ feed into the model by trait (CCS, TCH, Fibre)

Layers

+ Convolutional layer 1D, 3 kernel size, 1 stride;
+ RELU layer
+ Normlization layer
+ Maxpool layer
+ Convolutional layer 1D, 3 kernel size, 1 stride;
+ Relu
+ Maxpool
+ Convolutional layer 1D, 3 kernel size, 1 padding 1 stride
+ Linear 6519 -> 3000
+ Dropout(0.5)
+ Linear 3000 -> 1
=======
* Need to define costom layer 
>>>>>>> Stashed changes



**Note**

Semi-parametric reproducing kernel Hilbert space (RKHS) regression models have also been advocated as a potential alternative to capture non-additive effects in genomic selection. [[3]](#3) [[4]](#4); 

Degradation problem for DL model

**Long short-term memory** (**LSTM**)



Offspring selection:

> After selection in the offspring generation, offspring of parents with greater Mendelian sampling variance will show a greater within-family selection differential. This suggests that variation in the Mendelian sampling variance among potential parents can be used to accelerate response to recurrent selection, or to increase the probability of breeding a top-ranking individual or commercial variety [[8]](#8) [[9]](#9)

Simulation genome selection

> With simulation, for example, one can create a virtual sample of the gametes of a selection candidate and estimate the variance in the genomic EBVs (GEBV) of these gametes. Thus, for breeding schemes with an existing genomic reference population, a known linkage map, and the availability of phased genotypes, it only requires computing time to obtain the SD of the gametic GEBV for all selection candidates. [[10]](#10)





[Reference]

<a name="1">[1]</a>Garsmeur O, Droc G, Antonise R et al (2018) A mosaic monoploid reference sequence for the highly complex genome of sugarcane. Nat Commun. https://doi.org/10.1038/s41467-018-05051-5

<a name="2">[2]</a>Piperidis G, Piperidis N, D’Hont A (2010) Molecular cytogenetic investigation of chromosome composition and transmission in sugarcane. Mol Genet Genom 284:65–73. https://doi.org/10.1007/s00438-010-0546-3

<a name="3">[3]</a>Daniel Gianola, Rohan L Fernando, Alessandra Stella, Genomic-Assisted Prediction of Genetic Value With Semiparametric Procedures, *Genetics*, Volume 173, Issue 3, 1 July 2006, Pages 1761–1776, https://doi.org/10.1534/genetics.105.049510

<a name="4">[4]</a> Daniel Gianola, Johannes B C H M van Kaam, Reproducing Kernel Hilbert Spaces Regression Methods for Genomic Assisted Prediction of Quantitative Traits, *Genetics*, Volume 178, Issue 4, 1 April 2008, Pages 2289–2303, https://doi.org/10.1534/genetics.107.084285

<a name="5">[5]</a> Thirugnanasambandam PP, Hoang NV, Henry RJ. The Challenge of Analyzing the Sugarcane Genome. Front Plant Sci. 2018 May 14;9:616. doi: 10.3389/fpls.2018.00616. PMID: 29868072; PMCID: PMC5961476.

<a name="6">[6]</a>Zingaretti LM, Gezan SA, Ferrão LFV, Osorio LF, Monfort A, Muñoz PR, Whitaker VM, Pérez-Enciso M. Exploring Deep Learning for Complex Trait Genomic Prediction in Polyploid Outcrossing Species. Front Plant Sci. 2020 Feb 6;11:25. doi: 10.3389/fpls.2020.00025. PMID: 32117371; PMCID: PMC7015897.

<a name="7">[7]</a> Goddard, M. Genomic selection: prediction of accuracy and maximisation of long term response. *Genetica* **136,** 245–257 (2009). https://doi.org/10.1007/s10709-008-9308-0

<a name="8">[8]</a> Bernardo R. Genomewide selection of parental inbreds: classes of loci and virtual biparental populations[J]. Crop Science, 2014, 54(6): 2586-2595.

<a name="9">[9] </a>Segelke D, Reinhardt F, Liu Z, Thaller G. Prediction of expected genetic variation within groups of offspring for innovative mating schemes. Genet Sel Evol. 2014 Jul 2;46(1):42. doi: 10.1186/1297-9686-46-42. PMID: 24990472; PMCID: PMC4118311.

<a name="10">[10]</a> Bijma P, Wientjes YCJ, Calus MPL. Breeding Top Genotypes and Accelerating Response to Recurrent Selection by Selecting Parents with Greater Gametic Variance. *Genetics*. 2020;214(1):91-107. doi:10.1534/genetics.119.302643
