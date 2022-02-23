Month 7 Notebook
====

Date: 01/Feb/2021 - 28/Feb/2021

<br>
----



Goal:

+ create distribution plot for conv signals
+ sort and create GRM plot and subplot by regions
+ RF training by regions
+ AI - Sugarcane draft
+ VGG net
+ Res net
+ Boosting 

Read list:
+ https://www.biorxiv.org/content/10.1101/2020.09.15.276121v1.full (h2 description)
+ https://academic.oup.com/bfg/article/9/2/166/216335
+ https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0128570 (MNV)
+ https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9244164 (CNN feature importance)
+ Visualizing and Understanding Convolutional Networks **[ arXiv:1311.2901](https://arxiv.org/abs/1311.2901)**
  + Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. [Understanding neural networks through deep visualization](http://arxiv.org/abs/1506.06579). Presented at the Deep Learning Workshop, International Conference on Machine Learning (ICML), 2015.
+ https://github.com/yosinski/deep-visualization-toolbox
+ https://www.business-science.io/r/2022/02/22/my-4-most-important-explainable-ai-visualizations-modelstudio.html?utm_content=bufferb31e4 （visual method)

Import picard as RF save function 

Using Eric's R script to regenrate the GRM and sub-GRM

Identify reasons of different accuracy in region separated predictions 

	+ family (genomic corrlation) - maybe less family cluster(relationship) in Central subgroup
	+ 

perform region~ RF test (need over 36h)

(3 traits, 4 regions, 4*6 para sets, 10 rounds per set)



CNN feature importance:

method 1:

​	Top-K rank feature importance export:

		1. rank the clones by its certain trait (TCH, CCS, Fibre), get top 300(other) individuals
		1. sort the genotype map by the ranked index
		1. export the convoluted feature maps (multi-channels)
		1. save them separated by trait, channels, conv layer



A framework for CNN

![image-20220216153118905](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220216153118905.png)
