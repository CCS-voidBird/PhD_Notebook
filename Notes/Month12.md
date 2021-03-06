

todo list:

+ finish literature review
+ finish sugarcane ML draft editing 
+ Review of GxG winter school
+ Search and understand GibbsNet
+ create subset for test -- (csv format and ped format) (randomly 100 Clones, overall 1000 records with 1000 genotypes)
+ Functions to read plink files



literature review: 

[Genomic   predictions for enteric methane production are improved by metabolome and   microbiome data in sheep (Ovis aries). ](https://pubmed.ncbi.nlm.nih.gov/32815548/)  Ross EM, Hayes BJ, Tucker D, Bond J, Denman SE, Oddy VH.  J Anim Sci. 2020 Oct 1;98(10):skaa262. doi:  10.1093/jas/skaa262.  PMID: 32815548 Free PMC article. 

 [Metagenomic   predictions: from microbiome to complex health and environmental phenotypes   in humans and cattle. ](https://pubmed.ncbi.nlm.nih.gov/24023808/)  Ross EM, Moate PJ, Marett LC, Cocks BG, Hayes BJ.  PLoS One. 2013 Sep 4;8(9):e73056. doi:  10.1371/journal.pone.0073056. eCollection 2013.  PMID: 24023808 Free PMC article. 

Application of Bayesian genomic prediction methods to genome-wide association analyses | Genetics Selection Evolution https://gsejournal.biomedcentral.com/articles/10.1186/s12711-022-00724-8

Abdollahi-Arpanahi, R., Gianola, D. & Peñagaricano, F. Deep learning versus parametric and ensemble methods for genomic prediction of complex phenotypes. *Genet Sel Evol* **52,** 12 (2020). https://doi.org/10.1186/s12711-020-00531-z

Kemper KE, Bowman PJ, Pryce JE, Hayes BJ, Goddard ME. Long-term selection strategies for complex traits using high-density genetic markers. J Dairy Sci. 2012 Aug;95(8):4646-56. doi: 10.3168/jds.2011-5289. 

Dasari, C.M., Bhukya, R. Explainable deep neural networks for novel viral genome prediction. Appl Intell 52, 3002–3017 (2022). https://doi.org/10.1007/s10489-021-02572-3

CNN for saliency detection with low-level feature integration https://doi.org/10.1016/j.neucom.2016.11.056

Genomic simulation program:

MoBPS : Pook, T., Schlather, M., and Simianer, H. (2020). MoBPS - modular breeding program simulator. *G3* 10, 1915–1918. doi: 10.1534/g3.120.401193

Alzubaidi, L., Zhang, J., Humaidi, A.J. *et al.* Review of deep learning: concepts, CNN architectures, challenges, applications, future directions. *J Big Data* **8,** 53 (2021). https://doi.org/10.1186/s40537-021-00444-8



SVM review:
Applications of Support Vector Machine in Genomic Prediction in Pig and Maize Populations  https://doi.org/10.3389/fgene.2020.598318
Learning Interpretable SVMs for Biological Sequence Classification

**REVIEW GAN network and RNN network** 

http://dprogrammer.org/rnn-lstm-gru



Matrix analysis books:

- Matrix Analysis, Roger A. Horn, Charles R. Johnson
- Topics in Matrix Analysis, Roger A. Horn, Charles R. Johnson
- Matrix Analysis (Graduate Texts in Mathematics), Rajendra Bhatia
- Applied Linear Algebra and Matrix Analysis (Undergraduate Texts in Mathematics), Thomas S. Shores
- Linear Algebra Through Geometry, Thomas Banchoff and John Wermer





Running task：

+ CNN_2015_vs_2017 == log: CNN_17_all_4L.out [2022/6/2]



link: a 1D CNN explanation website

https://e2eml.school/convolution_one_d.html



U-Net 1D keras:

https://www.kaggle.com/code/kmat2019/u-net-1d-cnn-with-keras/notebook



Full equations of CNN backpropagation

https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b



Literature review 

A brief history of AI and machine learning (3-4 pages) 

- Why    it was developed, etc  
- Overview     of approaches – RF, NN 

Machine applied to crop and animal breeding 

- Gianola     etc.  

Most prospective applications of machine learning in crop and livestock breeding 

- Large     non-additive variation 
- Very     complex multi-trait interactions 
- Segment     stacking (Kemper et al. 2012). 



Workshop of quantative genetics

R.A Fisher - a large amount of mendelian genes 



Literature of multi-trait prediction

Multi-trait and multi-environment Bayesian analysis to predict the G x E interaction in flood-irrigated rice https://doi.org/10.1371/journal.pone.0259607

Use of multiple traits genomic prediction, genotype by environment interactions and spatial effect to improve prediction accuracy in yield data https://doi.org/10.1371/journal.pone.0232665

Multi-trait Genomic Selection Methods for Crop Improvement https://doi.org/10.1534/genetics.120.303305

Multi-Trait Genomic Prediction Improves Predictive Ability for Dry Matter Yield and Water-Soluble Carbohydrates in Perennial Ryegrass https://doi.org/10.3389/fpls.2020.01197

GxG Winter school - Review



GBLUP - SNPs and errors are normal distributed, every SNP contain a small effect to trait ,lead to all the SNP effects become extremely small

Bayesian Alphabet - SNPs contain 0 effect, and large SNP effects are allowed.

								 - calculation cost related to assumption distribution (functions) and need Gibbs sampling



GWAS - calculate SNP associations as significance 

SBLUP - Summary BLUP. use GWAS summary data and other LD informations

GBLUP/Bayesians with genetic functional annotation information

Non-additive effects - need review



Cholesky decomposition - from grm to data matrix 

Bayesian multiple-trait multiple-environment (BMTME) model



Y=Xβ+Z1b1+Z2b2+E

Pheno = Env(Trait~trait) + sum(genetic) + env(genetic) + residual
