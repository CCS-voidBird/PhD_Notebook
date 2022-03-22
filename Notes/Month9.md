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
+ Statistical methods for SNP heritability estimation and partition: A review https://doi.org/10.1016/j.csbj.2020.06.011

Check list:

UQ winter semester

AU summer semester (OPTIONAL)

gcta RMEL (rrBLUP)

Var SNP importance (Mark effects)





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



RF importance vs rrBLUP marker effects

1.SNPeffect^2 * freq_snp * (1-freq_snp) * 2 = Var_snps

2.Var_snp vs RF_importance 

PVE formula:

![img](https://pic3.zhimg.com/80/v2-a35e65ce1941e3de36f7239edb5632fa_720w.jpg)

TCH:

![image-20220316231556089](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220316231556089.png)

rrBLUP error solution * 

add diag pusedo value



create gcta required files:

```R
map <- data.frame(c(rep(0,dim(genos)[2]-1)))
map[,2] <- 1:(dim(genos)[2]-1)
map[,3:4] <- c(rep(0,dim(genos)[2]-1),rep(0,dim(genos)[2]-1))
write.table(as.matrix(map),file="./sugarcane.map",sep = "\t",col.names = F,row.names = F)


selected_set = train_set
sample = selected_set$Clone
size = dim(selected_set)
ped = data.frame(sample,sample,rep(0,size[1]),rep(0,size[1]),rep(0,size[1]),selected_set$TCHBlup)
dim(ped)
write.table(as.matrix(ped),file="./sugarcane.fam",sep = "\t",col.names = F,row.names = F)

phen = ped[,c(1,2,6)]
write.table(as.matrix(ped),file="./sugarcane.phen",sep = "\t",col.names = F,row.names = F)
```



gcta rrBLUP like code:

data preprocess:

train: 2013-2015, overall 22834 individuals, 1827 Clones

```R
dim(train_set) #22834 individual records
dim(raw_genos) #1827 Clones
library(dplyr)

#make plink required file
selected_set = train_set
sample = selected_set$Clone
size = dim(selected_set)

data.frame(
    sample,sample,rep(0,size[1]),rep(0,size[1]),rep(0,size[1]),selected_set$TCHBlup
)
# save as fam file 
write.table(ped,file="./sugarcane.fam",sep = "\t",col.names = F,row.names = F,quote = F)

# create phenotype file from fam variable
phen = ped[,c(1,2,6)]
dim(phen)
phen = cbind(phen,train_set$CCSBlup,train_set$FibreBlup)
write.table(phen,file="./sugarcane.phen",sep = "\t",col.names = F,row.names = F,quote = F)

# replicate Clone genotype data for ped file
colnames(ped)[2] = "Clone"
ped = left_join(ped,raw_genos,by="Clone")
save(ped,raw_genos,file="./tch_ped.RData")
dim(ped) # (22834,22086)
write.table(ped,file="./sugarcane.ped",sep = "\t",col.names = F,row.names = F,quote = F)

```

Plink codes:

```bash
plink --file sugarcane --geno 0.5 --out sugarcane_qc
```

gcta TCH REML

```bash
gcta64 --bfile sugarcane_qc --make-grm --out sugarcane_qc
gcta64 --reml --grm sugarcane_qc --pheno sugarcane.phen --mpheno 1 --reml-pred-rand --out sugarcane_tch
gcta64 --bfile sugarcane_tch --blup-snp sugarcane_tch.indi.blp --out test
```

![image-20220319195530421](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220319195530421.png)

> The GREML method uses REML for variance estimation (please see [Yang et al. 2010 AJHG](http://www.cell.com/ajhg/abstract/S0002-9297(10)00598-7) for details), which requires the inverse of the variance-covariance matrix **V**. If **V** is not positive definite, the inverse of **V** does not exist. We therefore could not estimate the variance component. This usually happens when one (or more) of the variance components are negative or constrained at zero. It might also indicate there is something wrong with the GRM or the data which you might need to check carefully.
>
> Unfortunately, there has not been an ultimate solution. Tricks such as adding a small number of to the diagonal elements of **V** also do not guarantee the modified **V** being invertible. In some cases, you might be able to get around the problem by using alternative REML algorithms e.g. the Fisher scoring approach (--reml-alg 1).
>
> We have implemented the "bending" approach ([Hayes and Hill 1981 Biometrics](http://www.jstor.org/stable/2530561?seq=1#page_scan_tab_contents)) in GCTA to invert **V** if **V** is not positive definite (you could add the --reml-bendV option to a REML or MLMA analysis to activate this approach). The "bending" approach guarantees to get an approximate of **V-1** but it does not guarantee the REML analysis being converged.
>
> **Note that the --reml-bendV option only provides an approximate inverse of \**V\** and has not been tested extensively. The results from analyses using this option might not be reliable.**



Currently using the alternative option "--reml-bendV" 

```bash
gcta64 --reml --grm sugarcane_qc --pheno sugarcane.phen --mpheno 1 --reml-bendV --reml-pred-rand --out sugarcane_tch --thread-num 24
```
REML with no constrain mode

```bash
gcta64 --reml --grm sugarcane_qc --pheno sugarcane_multi.phen --mpheno 1 --reml-no-constrain --reml-pred-rand --out sugarcane_tch --thread-num 24
```





ML/REML solve example

https://rh8liuqy.github.io/Example_Linear_mixed_model.html
