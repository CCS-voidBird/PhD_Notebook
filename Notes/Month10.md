Month 10 Notebook
====

Date: 01/Apr/2022 - 31/Apr/2022

reading list:

Yang, Jian et al. “GCTA: a tool for genome-wide complex trait analysis.” *American journal of human genetics* vol. 88,1 (2011): 76-82. doi:10.1016/j.ajhg.2010.11.011

https://scikit-learn.org/stable/modules/tree.html#tree

To DO List:

+ add stat methods and detail to sugarcane draft
+ milestone1 draft and checklist  
+ Draft edit: Comparison of STAT methods with valid set with best-test ML model 
+ https://zhuanlan.zhihu.com/p/43833351 History of Machine Learning



diag test in sugarcane_mean 

```bash
gawk '{if ($1 == $2) print $1,$2,$3,$4+0.1; else print $1,$2,$3,$4 }' sugarcane_mean_qc.grm | gzip > sugarcane_mean_qc_diag.grm.gz && \
gcta64 --reml --grm-gz sugarcane_mean_qc_diag --out test_piror --reml-pred-rand --pheno sugarcane_mean.phen --mpheno 3 --reml-alg 0 --reml-est-fix 
```



![image-20220407183140670](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20220407183140670.png)



Question: Lasso or ridge in GREML



Make Genotype data for valid set by plink - 

```bash
plink --file sugarcane_17 --recodeA --geno 0.5 --allow-no-sex --out sugarcane_17_a --thread-num 4
```



Norm CNN

for sample with N SNPs

active function $f_{act}$: $argmax(0,+\infin)$ 

$\hat{y} = \sum[w_i \times N_i \times \sum(\mathcal{N}(\mu_i ,\sigma^2_i,b_\Delta) )+b_i]$  

$N_i$ is certain SNP value (0,1,2 or -1,0,1) 

$\mathcal{N}(\mu_i ,\sigma^2_i,b_\Delta)$ is a function of norm distribution that 

$b_\Delta$ is a intercept on Y axis and control the overall area of norm function

