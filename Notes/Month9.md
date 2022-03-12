Month 9 Notebook
====

Date: 01/Mar/2022 - 31/Mar/2022



Goal:

+ Cor map for RF importance and rr-BLUP marker effects
+ Edit sugarcane draft
+ Milestone 1 draft
+ solve bug in rrblup - https://github.com/cran/rrBLUP/blob/master/R/mixed.solve.R
+ VGG, RNN, GRU, LSTM model (structure)

Reading

+ http://www.seas.ucla.edu/~vandenbe/133B/lectures/psd.pdf semi-positive definite
+ https://pbgworks.org/sites/pbgworks.org/files/Introduction%20to%20Genomic%20Selection%20in%20R.pdf rrBLUP usage



GBLUP - GRM parameter:

```R
#add parameter "max.missing" that set a threshold for filling missing data. While a SNP value is NA in over 50% Clones, this col will remain NA,and should be filtered.
g.d <- A.mat(pure_genos,impute.method = "mean",return.imputed = T)
g.d <- A.mat(pure_genos,impute.method = "EM",return.imputed = T)
imputed_geno <- g.d$imputed
#test if imputed genos contain NA value (The col has over 50% of lines with missed data)
NA %in% imputed_geno
#Remove cols with NA values
library(dplyr)
imputed_geno <- as.data.frame(imputed_geno) %>% select_if(~any(!is.na(.)))
#Check changes inside the filtered genos
dim(imputed_geno)[2] == dim(pure_genos)[2]
#True
#rr BLUP
rr.tch <- mixed.solve(y=train_set$TCHBlup,Z=rr_genos[,2:l])
rr.ccs <- mixed.solve(y=train_set$CCSBlup,Z=rr_genos[,2:l])
rr.tch <- mixed.solve(y=train_set$FibreBlup,Z=rr_genos[,2:l])
#GBLUP
g.tch <- kin.blup(data=train_set,geno="Clone",pheno="TCHBlup",K=g.d$A)
g.ccs <- kin.blup(data=train_set,geno="Clone",pheno="CCSBlup",K=g.d$A)
g.fibre <- kin.blup(data=train_set,geno="Clone",pheno="F",K=g.d$A)
```

Additional for EM impute method:

While in the higher marker density, the GRM is estimated as $$A=WW^{'}/c$$, where $W_{ik} = X_{ik} + 1 -2p_k$ and $p_k$ is the frequency of the 1allele at marker k. By using a normalization constant of $c=2\sum_kp_k(1-p_k)$, the mean of the diagonal elements is $1 + f$   

The EM imputation algorithm is based on the multivariate normal distribution and was designed for
use with GBS (genotyping-by-sequencing) markers, which tend to be high density but with lots of
missing data. Details are given in Poland et al. (2012). The EM algorithm stops at iteration t when
the RMS error = $n^{-1}||A_t = A_{t-1}||_2 < tol$ 

Shrinkage estimation can improve the accuracy of genome-wide marker-assisted selection, partic-
ularly at low marker density (Endelman and Jannink 2012)

**rrBLUP bug record:**

K should be semi-positive definite 





Conv1 -> env1 = hidden 1

Conv1 -> conv2 -> env2 = hidden 2

(hidden1 + hidden2) -> env3 



env layer : {e1,e2,e3...et} vector

e: 
