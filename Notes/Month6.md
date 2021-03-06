Month 6 Notebook
====

Date: 01/Dec/2021 - 31/Dec/2021

<br> Editor: Chensong Chen
----

Current Goals:

+ Write a report for current results
+ import keras API for reading data and making a mixed model
+ optimize/solve structural issues described by the **Navigating the pitfalls of applying machine learning in genomics**
+ Add runtime record function to programs
+ Try fastAI
+ import DeepGS model in a previous study
+ Understanding MNV: mean normalized discounted cumulative gain value
+ 3D heatmap for convolutional output (markers, clones, channels) （Jan）
+ Export decision tree for marker importance in RF （Jan）https://towardsdatascience.com/4-ways-to-visualize-individual-decision-trees-in-a-random-forest-7a9beda1d1b7
+ add earlystopping methods (DONE)
+ Add a increase rate recording function for CNNs (instead 500 epoch)



Literature list:

+ Whalen, S., Schreiber, J., Noble, W.S. et al. Navigating the pitfalls of applying machine learning in genomics. Nat Rev Genet (2021). https://doi.org/10.1038/s41576-021-00434-9
+ Deomano E, Jakson P, Wei X, Aitken K, Kota R, Perez-Rodriguez P (2020) Genomic Prediction of sugar content and cane yield in sugar cane clones in different stages of selection in a breeding program, with and without pedigree information. Mol Breed. https://doi.org/10.1007/s11032-020-01120-0
+ Ma, W., Qiu, Z., Song, J. *et al.* A deep convolutional neural network approach for predicting phenotypes from genotypes. *Planta* **248,** 1307–1318 (2018). https://doi.org/10.1007/s00425-018-2976-9
+ Yadav, Seema, et al. "Accelerating genetic gain in sugarcane breeding using genomic selection." *Agronomy* 10.4 (2020): 585.





[Whalen, S., Schreiber, J., Noble, W.S. et al. Navigating the pitfalls of applying machine learning in genomics. Nat Rev Genet (2021). https://doi.org/10.1038/s41576-021-00434-9]
Traps in genetic ML:

+ Distributional differences
+ dependent examples
+ confounding
+ leaky preprocessing
+ unbalanced classes



RF performance from n_features 10-5000:

![rm_comp_10_5000.png](https://github.com/CCS-voidBird/PhD_Notebook/blob/main/pic/2013to15vs2017/rm_comp_10_5000.png?raw=true)

New 1D CNN: OHE on

![1DCNN_onehot.png](https://github.com/CCS-voidBird/PhD_Notebook/blob/main/pic/2013to15vs2017/1DCNN_onehot.png?raw=true)

Accuracy comparison by various decision trees: max 1000 features per tree

![RF_n_tree.png](https://github.com/CCS-voidBird/PhD_Notebook/blob/main/pic/RF_n_tree.png?raw=true)
