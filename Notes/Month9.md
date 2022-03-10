Month 9 Notebook
====

Date: 01/Mar/2022 - 31/Mar/2022



Goal:

+ Cor map for RF importance and rr-BLUP marker effects
+ Edit sugarcane draft
+ Milestone 1 draft

Reading

+ http://www.seas.ucla.edu/~vandenbe/133B/lectures/psd.pdf semi-positive definite
+ https://pbgworks.org/sites/pbgworks.org/files/Introduction%20to%20Genomic%20Selection%20in%20R.pdf rrBLUP usage



GBLUP - GRM parameter:

```R
#add parameter "max.missing" that set a threshold for filling missing data. While a SNP value is NA in over 50% Clones, this col will remain NA,and should be filtered.
g.d <- A.mat(pure_genos,max.missing = 0.5,impute.method = "mean",return.imputed = T)
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

