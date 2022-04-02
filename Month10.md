Month 10 Notebook
====

Date: 01/Apr/2022 - 31/Apr/2022

reading list:

Yang, Jian et al. “GCTA: a tool for genome-wide complex trait analysis.” *American journal of human genetics* vol. 88,1 (2011): 76-82. doi:10.1016/j.ajhg.2010.11.011

diag test in sugarcane_mean 

```bash
gawk '{if ($1 == $2) print $1,$2,$3,$4+0.1; else print $1,$2,$3,$4 }' sugarcane_mean_qc.grm | gzip > sugarcane_mean_qc_diag.grm.gz && \
gcta64 --reml --grm-gz sugarcane_mean_qc_diag --out test_piror --reml-pred-rand --pheno sugarcane_mean.phen --mpheno 3 --reml-alg 0 --reml-est-fix 
```

