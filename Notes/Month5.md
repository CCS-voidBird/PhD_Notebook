Month 5 Notebook
====

Date: 01/Nov/2021 - 30/Nov/2021

<br> Editor: Chensong Chen
----

Current Goals:

+ preparing QSA presentation (due 22 Nov)
+ Testing some ideas in CNN model 
  + repeating non-genetic factors as a independent vector, contributing them with genotypes as a 2D matrix
  + hamming distance
+ touch Random forest model
+ Confirmation index
+ Try a MLP model (Sandhu Km et)
+ Review Seema's sugarcane paper to format the data pre-process (e.g. distribution transformation)
+ Try run RF by less n_features (~50)

Literature List:

+ Sandhu, K. S., Lozada, D. N., Zhang, Z., Pumphrey, M. O., & Carter, A. H. (2021a). Deep learning for predicting complex traits in spring wheat breeding program. *Frontiers in Plant Science*, 11, 613325. https://doi.org/10.3389/fpls.2020.613325
+ Wang, T., Chen, YP.P., Goddard, M.E. *et al.* A computationally efficient algorithm for genomic prediction using a Bayesian model. *Genet Sel Evol* **47,** 34 (2015). https://doi.org/10.1186/s12711-014-0082-4
+ Engle BN, Corbet NJ, Allen JM, Laing AR, Fordyce G, McGowan MR, Burns BM, Lyons RE, Hayes BJ. Multivariate genomic predictions for age at puberty in tropically adapted beef heifers. J Anim Sci. 2019 Jan 1;97(1):90-100. doi: 10.1093/jas/sky428. PMID: 30481306; PMCID: PMC6313118.
+ Stephan, J., Stegle, O. & Beyer, A. A random forest approach to capture genetic effects in the presence of population structure. *Nat Commun* **6,** 7432 (2015). https://doi.org/10.1038/ncomms8432
+ Whalen, S., Schreiber, J., Noble, W.S. *et al.* Navigating the pitfalls of applying machine learning in genomics. *Nat Rev Genet* (2021). https://doi.org/10.1038/s41576-021-00434-9

>  **The prediction accuracy was calculated as the correlation between the genomic estimated breeding values and the residual phenotype (phenotype adjusted for fixed effects using linear modeling) and divided by the square root of the estimated heritability.**



A online convolutional layer explainer:

https://poloclub.github.io/cnn-explainer/



Experiment in RM/MLP/CNN

[BASIC]
traits = TCHBlup#CCSBlup#FibreBlup
drop = Trial#Crop#Clone
OneHot = 1

[CNN]
lr = 0.0001
n_layers = 8
n_units = 5,10,15,25

[TDCNN]
lr = 0.0001
n_layers = 8
n_units = 5,10,15,25

[MLP]
lr = 0.0001
n_layers = 4,8,16
n_units = 10,15,25,45

[RM]
n_estimators = 200
max_features = 10,25,35,50,100,300,500,1000,2000,5000



Current outcomes:

TCH: accuracy increased by n_units 



![method_comp_plus_rm.png](https://github.com/CCS-voidBird/PhD_Notebook/blob/main/pic/method_comp_plus_rm.png?raw=true)



Next stage: testing hp: n_layers

boosting methods

Regression model accuracy metrics: R R-square, (AIC, BIC) - choose R





BUG/ISSUES:（SOLVED）

Clones overlapping between genos and phenos  = 23, overall 227 records 

overall overlapped Clones between genos and phenos = 42



Discussion of Train-Test-Valid set selection:



1. selecting by certain non-genetic factors (Series, Region) such as 1st factor, 2nd factor. The 1st factor will be used to determine processing set and valid set; the 2nd factor will be used inside the processing set, and separate into test set and valid set



The assumptions of ML, DL

> It **assumes that there is minimal or no multicollinearity among the independent variables**. It usually requires a large sample size to predict properly. It assumes the observations to be independent of each other.



Traps in genetic ML:

+ Distributional differences
+ dependent examples
+ confounding
+ leaky preprocessing
+ unbalanced classes





###################################################################################

#########################INDEX#################################################

###################################################################################

1. Project Summary
   1. From background to research aims
   2. The approaches will be used in research activations
   3. The likely significance of the research
2. Research Background
   1. Refer to reliable previous researches (literature), identify at least one blank/gap in certain field.
   2. Assign a aim - Answering for a question; Hypothesis; Achieve a creative goal.
3. Research approach
   1. Genomic selection
   2. Artificial intelligence 
   3. Biological regression
4. Significant of the Research
   1. Describe the contribution that the research will achieve.
   2. Provide information, insights, potential applications or direct material outcomes 
5. Timeline
   1. The timeline that probably estimate the time of various research activities.
   2. Include: writing thesis chapters, thesis submission, Relative conference meeting and training. 
   3. Format: diagrammatic workplan - flowchart, Gantt chart
6. Thesis Outline
   1. The structure of the thesis
7. Additional Resources and Training
   1. Any requirements, software, database access, special training
8. Budget
9. Reference

###################################################################################

################################Research Background###################################

###################################################################################

Agricultural breeding 
