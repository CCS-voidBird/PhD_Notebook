Week1-2 Notebook
====

Date: 19/July/2021 - 3/Aug/2021
<br> Editor: Chensong Chen
----

Current Goals:
1. Rerunning Kira's SimulationGenome software. [70%] 
2. Literature review
    + [DONE] Montesinos-López OA, Montesinos-López A, Pérez-Rodríguez P, Barrón-López JA, Martini JWR, Fajardo-Flores SB, Gaytan-Lugo LS, Santana-Mancilla PC, Crossa J. A review of deep learning applications for genomic selection. BMC Genomics. 2021 Jan 6;22(1):19. doi: 10.1186/s12864-020-07319-x. 
    + [DONE] Abdollahi-Arpanahi R, Gianola D, Peñagaricano F. Deep learning versus parametric and ensemble methods for genomic prediction of complex phenotypes. Genet Sel Evol. 2020 Feb 24;52(1):12. doi: 10.1186/s12711-020-00531-z
    + Kemper KE, Bowman PJ, Pryce JE, Hayes BJ, Goddard ME. Long-term selection strategies for complex traits using high-density genetic markers. J Dairy Sci. 2012 Aug;95(8):4646-56. doi: 10.3168/jds.2011-5289. 
    + Bijma, P., Wientjes, Y.C.J., Calus, M.P.L.  Breeding top genotypes and accelerating response to recurrent selection by selecting parents with greater gametic variance (2020) Genetics, 214 (1), pp. 91-107. DOI: 10.1534/genetics.119.302643
    + Hickey, L.T., N. Hafeez, A., Robinson, H., Jackson, S.A., Leal-Bertioli, S.C.M., Tester, M., Gao, C., Godwin, I.D., Hayes, B.J., Wulff, B.B.H. Breeding crops to feed 10 billion (2019) Nature Biotechnology, 37 (7), pp. 744-754. Cited 133 times. DOI: 10.1038/s41587-019-0152-9
    + Dias, R., Torkamani, A. Artificial intelligence in clinical and genomic diagnostics. Genome Med 11, 70 (2019). https://doi.org/10.1186/s13073-019-0689-8




Three statistical methods for predicting phenotypes from dense SNP markers.

+ BLUP/GBLUP/GS-BLUP
+ BayesA
+ BayesB



Notes:
1. Descriptions from Kira's message:
> GA inputs:  
>+ params.txt (Parameters: used to tell the GA the size of the segment_effects.txt matrix, how many lines it should select, how many generations to run for, etc. The R script I sent you modifies this before running the GA.)
>+ segment_effects.txt (rows = haplotypes of simulated lines, columns = local GEBVs for a particular block. The R script I sent you creates this.)
>+ iseed.txt (you can change the random number generator seed using this. I don't usually bother.)

> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.

2. Notes from literatures:
    + Today, genomic selection (GS), proposed by Bernardo [[1]](#1) and Meuwissen et al. [[2]](#2) has become an established methodology in breeding.
    + Comparing to phenotypic selection (PS) :

        |Species|Condition |Label | GS | PS |
        |---------|--------|------|----|----|
        |maize [[3]](#3)|*Drought condition* | Gain (t/ha)  | 0.50 | 0.27 |
        |     |                    | Genetic Gain | 0.124 | 0.067 |
        |     |*Optimal condition* | Gain (t/ha)  | 0.55 | 0.34 |
        |     |                    | Genetic Gain | 0.140 | 0.084 |
        |maize [[4]](#4)| - | selection gain | similar ||
        |soybean [[5]](#5)| - | fatty acid traits | higher | - |
        |                 | - | yield, protein, oil | similar |
        |barley [[6]](#6) | - | selection gain | similar ||
        |                 | - | breeding cycle | shorter | - |
        |                 | - | costs | lower | - |
        ||
        
    + genomic best linear unbiased prediction (GBLUP) [[7]](#7).
    + DL is a type of machine learning (ML) approach that is a subfield of artificial intelligence (AI). The main difference between DL methods and conventional statistical learning methods is that DL methods are nonparametric models providing tremendous flexibility to adapt to complicated associations between data and output [[7]](#7).
    + Pros of DL while vs ML [[8]](#8):
        + Stronger ability of discovering hidden patterns.
    + Deep neural network
        + Activation function
            + Non-linear
            + Continuously differentiable
            + Fixed Range
            
        + Loss function
          
        + Feedforward networks (or multilayer perceptrons; MLPs) (MLP)
            + developing step by step (one layer to next level layer);
            + no super links that cross layers;
            + basic and relatively easy to train;
            + has risks of overfitting. 
            
        + Recurrent neural networks (RNN)
            + a<sup>\<t\></sup> = g<sub>1</sub>(W<sub>aa</sub>a<sup>\<t-1\></sup> + W<sub>ax</sub>X<sup>\<t\></sup>+b<sub>a</sub>)
            + y<sup>\<t\></sup> = g<sub>2</sub>(W<sub>ya</sub>a<sup>\<t-1\></sup> + b<sub>y</sub>)
                + W: Weight;
                + W<sub>aa</sub>: Weight of previous activations.
                + W<sub>ax</sub>: Weight of activation of current X (input).
                + b: background (?) [[9]](#9)
            + Road map [[9]](#9)
            
            ![Foluma image](https://stanford.edu/~shervine/teaching/cs-230/illustrations/architecture-rnn-ltr.png?9ea4417fc145b9346a3e288801dbdfdc)
            + Inside system [[9]](#9)
            ![RNN image](https://stanford.edu/~shervine/teaching/cs-230/illustrations/description-block-rnn-ltr.png?74e25518f882f8758439bcb3637715e5)
            
            + Common activation functions[[9]](#9): 
            
            |Sigmoid|Tanh|RELU|
            |---|---|---|
            |![for1](https://latex.codecogs.com/gif.latex?g%28z%29%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-z%7D%7D)|![for2](https://latex.codecogs.com/gif.latex?g%28z%29%3D%5Cfrac%7Be%5E%7Bz%7D-e%5E%7B-z%7D%7D%7Be%5E%7Bz%7D&plus;e%5E%7B-z%7D%7D)|![for3](https://latex.codecogs.com/gif.latex?g%28z%29%3D%5Cmax%280%2Cz%29)|
            |![Activation functions](https://stanford.edu/~shervine/teaching/cs-229/illustrations/tanh.png?22ac27f27c510c6414e8a3bb4aca2d80)|![pic2](https://stanford.edu/~shervine/teaching/cs-229/illustrations/tanh.png?22ac27f27c510c6414e8a3bb4aca2d80)|![pic3](https://stanford.edu/~shervine/teaching/cs-229/illustrations/relu.png?6c1d78551355db5c6e4f6f8b5282cfa8)|
            ||
            + Multiple developing directions. (forward or backward);
            + ongoing connections leading to all the neurons in the subsequent layer. (current layer connects all 1-previous level layers)
            + recurrent connections that propagate information between neurons of the same layer.
            
        + Convolutional neural networks (CNN) [Next]
        
            + Convolutional layers
              + Input: tensor with shape.
              + A convolutional filters (kernels)
              + input/output channels (should be equal)(?)
              + hyper-parameters -- padding; stride and dilation (?)
            + Pooling layers
              + Functions: merge/split/reformat data of convolutional layers into other format/subtypes
            + Main solutions
              + Convolution, nonlinear transformation and pooling


[111](#test)



+ Deep learning nerual network tools: PyTorch [[10]](#10) Chainer [[11]](#11)
+ Note [[7]](#7): 

> Pérez-Rodríguez et al. [[12]](#12) compares the predictive ability of Radial Basis Function Neural Networks and Bayesian Regularized Neural Networks against several linear models [BL, BayesA, BayesB, BRR and semi-parametric models based on Kernels (Reproducing Kernel Hilbert Spaces)]. 

In prediction accuracy: Non-linear models performed better than the linear regression specification.

Some expermental records showed accuracy similarities in DL&linear models (e.g. GBLUP)  comparisons. 

+ Probabilistic neural network (PNN) has a relatively higher performance than MLP [[13]](#13)

Bayesian A/B testing [?]

+ CNN phenotype prediction case study [[14]](#14)

multi-trait deep learning (MTDL) vs Bayesian multi-trait and multi-environment (BMTME), accuracy comparison

Specific condition: genotype × environment interaction

| Species       | Model | imported | non-imported |
| ------------- | ----- | -------- | ------------ |
| maize         | BMTME | 0.456    | 0.317        |
|               | MTDL  | 0.407    | 0.435        |
| wheat         | BMTME | 0.812    | 0.765        |
|               | MTDL  | 0.759    | 0.876        |
| Iranian wheat | BMTME | 0.999    | 0.54         |
|               | MTDL  | 0.836    | 0.669        |

+ Univariate DL vs Support vector machine (SVM) vs TGBLUP* [[15]](#15)
  + *Conventional Bayesian threshold best linear unbiased prediction [?]

1. No siginificant differences in total.

2. Specifically (human case):

   Interaction imported: TGBLUP > DL > SVM;

   interaction excluded: no significant differences.

+ univariate DL (UDL) vs multi-trait deep learning (MTDL) vs GBLUP [[16]](#16)

1. Accuracy: GBLUP ≈ MTDL > UDL



Pros of DL [[7]](#7):

+ Naturally capture - free for definating additional relations sepcifically.
+ Efficiency and pre-processing free.
+ workflow flexible
+ outstanding for prediction based on a large dateset

Cons:

+ generated parameters with less biological meanings (in Bioscience)
+ Need a large amount of traing time. 
+ challenging in selecting hyper-parameters.



> For this reason, CNNs are being very successfully applied to complex tasks in plant science for: (a) root and shoot feature identification [[94](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-07319-x#ref-CR94)], (b) leaf counting [[95](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-07319-x#ref-CR95), [96](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-07319-x#ref-CR96)], (c) classification of biotic and abiotic stress [[97](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-07319-x#ref-CR97)], (d) counting seeds per pot [[98](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-07319-x#ref-CR98)], (e) detecting wheat spikes [[99](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-07319-x#ref-CR99)], and (f) estimating plant morphology and developmental stages [[100](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-020-07319-x#ref-CR100)], etc. 



**A study of comparing predictive performances of MLP and CNN** [[17]](#17)

> Moreover, interaction between loci is prevalent and recombination hotspots are not uniformly distributed across the genome. Some advanced machine-learning algorithms such as ensemble methods and deep learning (DL) algorithms might help in genome-enabled prediction.

For algorithm usage proposal 

> Boosting and RF are model specification free and may account for non-additive effects.Moreover, they are fast algorithms, even when handling a large number of covariates and interactions and can be used in both classification and regression problems.



Defination:

+ purely/combination additive &rarr; performed as a sum of all gonotype alleles.

+ non-additive gene action &rarr; only have one certain phenotype and have no additive action  (two-locus model)

  

A brife solution of Random forest

1. For sample data which contains N samples, select & return single sample for N times &rarr; a sub N sample data for a new decision tree.
2. For each sample that has M features, select m features (m << M) as the sub-feature in certain decision tree.
3. merge /weight decisions from the forest, got results.



Convolutional nerual network (CNN) case study in GS field [[17]](#17).

+ Input: genotype matrix of 80000 animals

1. Structure:
   + 1 input layer;
   + 1 convolutional layer (16 filters; 1x5 window size; 1x3 stride size) 
   + 1 max-pooling layer (1x2 window size; 1x2 stride size)
   + 2 full-connect layers (MLP) (32, 1 unit)
   + 2 dropout layers (0.3 dropping rate)
   + 1 output layer
2. Hyperparameters
   + Epochs: 200
   + Batch size: 64
   + Learning rate: 0.01
   + Momentum: 0.5 [?]
   + Weight decay: 0.00001
3. Activation Function
   + ReLu for convolutional layer;
   + Softrelu for 1st fill-connect layer;
   + linear activation function for the output layer;
4. Environment：DeepGS [[18]](#18) (R language/environment, version 3.6.1)

> We compared learning machines using two different types of predictor variables: (i) genotypes at causal loci, and (ii) genotypes at SNPs. In the former case, statistical methods were fitted using the genotypes at causal variants as predictors. In the latter case, to mimic the real SNP data, QTN were excluded from the genotypic matrix and genomic prediction was performed using only the genotypes at SNPs.





Discussion:



> Whole-genome prediction differs in a very important way from image or speech recognition tasks [[33](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-020-00531-z#ref-CR33)]. Complex traits are multifactorial, where environmental factors may differ from individual to individual, and epigenetic marks can affect performance, so that the genotype of an individual may not provide sufficient information to predict phenotypes accurately [[48](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-020-00531-z#ref-CR48)]. However, there are some similarities between genomics and other domains, for instance genotype–phenotype associations can be viewed as a landscape. This landscape may have extremely steep valleys, where small perturbations in genotype give rise to vastly different phenotypes [[49](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-020-00531-z#ref-CR49)]. It may also have large plateaus, where seemingly unrelated genotypes yield an equivalent phenotype.

[Link location]

[33] Goodfellow I, Bengio Y, Courville A. Deep learning. Cambridge: The MIT press; 2016.

[48] Leung MKK, Delong A, Alipanahi B, Frey BJ. Machine learning in genomic medicine: a review of computational problems and data sets. Proc IEEE. 2016;104:176–97.

[59] Hart JR, Zhang Y, Liao L, Ueno L, Du L, Jonkers M, et al. The butterfly effect in cancer: a single base mutation can remodel the cell. Proc Natl Acad Sci USA. 2015;112:1131–6.





**IDEA: Generative Adversarial Network for genomic selection (selecting parent pairs)**



**Question** for DNN: How to select/optimize b (bias)



Reference:

<a name="1">[1]</a> Bernardo, R. (1994), Prediction of Maize Single-Cross Performance Using RFLPs and Information from Related Hybrids. Crop Science, 34: 20-25 cropsci1994.0011183X003400010003x. https://doi.org/10.2135/cropsci1994.0011183X003400010003x

<a name="2">[2]</a> Meuwissen TH, Hayes BJ, Goddard ME. Prediction of total genetic value using genome-wide dense marker maps. Genetics. 2001 Apr;157(4):1819-29. PMID: 11290733; PMCID: PMC1461589.

<a name="3">[3]</a> Vivek BS, et al. Use of genomic estimated breeding values results in rapid genetic gains for drought tolerance in maize. Plant Genome. 2017;10:1–8.

<a name="4">[4]</a> Môro GV, Santos MF, de Souza Júnior CL. Comparison of genome-wide and phenotypic selection indices in maize. Euphytica. 2019;215:76. https://doi.org/10.1007/s10681-019-2401-x.

<a name="5">[5]</a> Smallwood CJ, Saxton AM, Gillman JD, Bhandari HS, Wadl PA, Fallen BD, Hyten DL, Song Q, Pantalone VR. Context-specific Genomic Selection Strategies Outperform Phenotypic Selection for Soybean Quantitative Traits in the Progeny Row Stage. Crop Sci. 2019;59(1):54–67.

<a name="6">[6]</a> Salam A, Smith KP. Genomic selection performs similarly to phenotypic selection in barley. Crop Sci. 2016;56(6):2871–2881.

<a name="7">[7]</a> Montesinos-López OA, Montesinos-López A, Pérez-Rodríguez P, Barrón-López JA, Martini JWR, Fajardo-Flores SB, Gaytan-Lugo LS, Santana-Mancilla PC, Crossa J. A review of deep learning applications for genomic selection. BMC Genomics. 2021 Jan 6;22(1):19. doi: 10.1186/s12864-020-07319-x. 

<a name="8">[8]</a> Kononenko I, Kukar M. Machine Learning and Data Mining: Introduction to Principles and Algorithms. London: Horwood Publishing; 2007.

<a name="9">[9]</a> https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks

<a name="10">[10]</a> Team, P.C. (2017). Pytorch: tensors and dynamic neural networks in Python with strong GPU acceleration. GitHub pub online: March 22, 2017. https://github.com/pytorch/pytorch.

<a name="11">[11]</a> Tokui, Seiya, et al. "Chainer: a next-generation open source framework for deep learning." *Proceedings of workshop on machine learning systems (LearningSys) in the twenty-ninth annual conference on neural information processing systems (NIPS)*. Vol. 5. 2015.

<a name="12">[12]</a> Pérez-Rodríguez P, Gianola D, González-Camacho JM, Crossa J, Manès Y, Dreisigacker S. Comparison between linear and non-parametric regression models for genome-enabled prediction in wheat. G3 (Bethesda). 2012;2(12):1595–605. https://doi.org/10.1534/g3.112.003665.

<a name="13">[13]</a> González-Camacho JM, Crossa J, Pérez-Rodríguez P, et al. Genome-enabled prediction using probabilistic neural network classifiers. BMC Genomics. 2016;17:1–16. https://doi.org/10.1186/s12864-016-2553-1.

<a name="14">[14]</a> Ma W, Qiu Z, Song J, Li J, Cheng Q, Zhai J, et al. A deep convolutional neural network approach for predicting phenotypes from genotypes. Planta. 2018;248:1307–18. https://doi.org/10.1007/s00425-018-2976-9.

<a name="15">[15]</a> Montesinos-López OA, Vallejo M, Crossa J, Gianola D, Hernández-Suárez CM, Montesinos-López A, Juliana P, Singh R. A benchmarking between deep learning, support vector machine and Bayesian threshold best linear unbiased prediction for predicting ordinal traits in plant breeding. G3: Genes Genomes Genetics. 2019a;9(2):601–18.

<a name="16">[16]</a> Montesinos-López OA, Montesinos-López A, Tuberosa R, Maccaferri M, Sciara G, Ammar K, Crossa J. Multi-trait, multi-environment genomic prediction of durum wheat with genomic best linear unbiased predictor and deep learning methods. Front Plant Sci. 2019;11(10):1–12.

<a name="17">[17]</a> Abdollahi-Arpanahi R, Gianola D, Peñagaricano F. Deep learning versus parametric and ensemble methods for genomic prediction of complex phenotypes. Genet Sel Evol. 2020 Feb 24;52(1):12. doi: 10.1186/s12711-020-00531-z

<a name="18">[18]</a> Ma W, Qiu Z, Song J, Li J, Cheng Q, Zhai J, et al. A deep convolutional neural network approach for predicting phenotypes from genotypes. Planta. 2018;248:1307–18.

<ins>222</ins>

<u>222</u>

<del>222</del>

&rarr;

&ensp;



