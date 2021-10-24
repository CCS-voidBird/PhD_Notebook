Month 4 Notebook
====

Date: 01/Oct/2021 - 31/Oct/2021

<br> Editor: Chensong Chen
----

Current Goals:

+ Preparing training matrix
+ Write a proposal 
+ add a function to model that receive multiple years (DONE)
+ add a OneHotEncoding function to model (to_categorical testing)
+ Rerun the model (2013-15 vs 2017)DONE

**continue: https://pubmed.ncbi.nlm.nih.gov/18076469/**

https://www.nature.com/articles/nrg2575 #pig

https://www.biorxiv.org/content/10.1101/2021.05.20.445038v1.full.pdf

Random forest example

https://redditech.github.io/team-fast-tabulous/jupyter/2021/07/21/Regressor-Versus-Classifier.html

Add to Read list:

+ https://github.com/ne1s0n/coding_excercises

Li, S., Yang, G., Yang, S. *et al.* The development of a high-density genetic map significantly improves the quality of reference genome assemblies for rose. *Sci Rep* **9,** 5985 (2019). https://doi.org/10.1038/s41598-019-42428-y

[Done] Bourke PM, Voorrips RE, Visser RGF, Maliepaard C. Tools for Genetic Studies in Experimental Populations of Polyploids. Front Plant Sci. 2018 Apr 18;9:513. doi: 10.3389/fpls.2018.00513. PMID: 29720992; PMCID: PMC5915555.

Biemans F, de Jong MCM, Bijma P. Genetic parameters and genomic breeding values for digital dermatitis in Holstein Friesian dairy cattle: host susceptibility, infectivity and the basic reproduction ratio. Genet Sel Evol. 2019 Nov 20;51(1):67. doi: 10.1186/s12711-019-0505-3. PMID: 31747869; PMCID: PMC6865030.

Santos DJA, Cole JB, Lawlor TJ Jr, VanRaden PM, Tonhati H, Ma L. Variance of gametic diversity and its application in selection programs. J Dairy Sci. 2019 Jun;102(6):5279-5294. doi: 10.3168/jds.2018-15971. Epub 2019 Apr 10. PMID: 30981488.

**T H E Meuwissen, B J Hayes, M E Goddard, Prediction of Total Genetic Value Using Genome-Wide Dense Marker Maps, *Genetics*, Volume 157, Issue 4, 1 April 2001, Pages 1819–1829, https://doi.org/10.1093/genetics/157.4.1819**

Goddard, M. Genomic selection: prediction of accuracy and maximisation of long term response. *Genetica* **136,** 245–257 (2009). https://doi.org/10.1007/s10709-008-9308-0

Khatkar, Mehar S., et al. "Quantitative trait loci mapping in dairy cattle: review and meta-analysis." *Genetics Selection Evolution* 36.2 (2004): 163-190.

Kendziorski, C., Wang, P. A review of statistical methods for expression quantitative trait loci mapping. *Mamm Genome* **17,** 509–517 (2006). https://doi.org/10.1007/s00335-005-0189-6

Goddard, M. and Hayes, B. (2007), Genomic selection. Journal of Animal Breeding and Genetics, 124: 323-330. https://doi.org/10.1111/j.1439-0388.2007.00702.x

polyploid related

(1) polyploid genotyping, including the scoring of marker dosage (allele counts) and generation of haplotypes; 

+ identifying single nucleotide polymorphisms (SNPs) - targeted SNP arrays

+ untargeted genotyping, usually directly sequencing from species - exome sequencing [1], target enrichment [2]

+ Comparison:

  + > The disadvantages of targeted approaches have been well explored (particularly regarding ascertainment bias, where the set of targeted SNPs on an array poorly represents the diversity in the samples under investigation due to biased methods of SNP discovery) ([Albrechtsen et al., 2010](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B3); [Moragues et al., 2010](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B103); [Didion et al., 2012](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B41); [Lachance and Tishkoff, 2013](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B87)), although there are advantages and disadvantages to both methods ([Mason et al., 2017](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B94)). Apart from costs, differences exist in the ease of data analysis following genotyping, with sequencing data requiring greater curation and bioinformatics skills ([Spindel et al., 2013](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B130); [Bajgain et al., 2016](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B6)) as well as potentially containing more erroneous and missing data ([Spindel et al., 2013](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B130); [Jones et al., 2017](https://www.frontiersin.org/articles/10.3389/fpls.2018.00513/full#B76)).	

(2) genetic and physical mapping, where we look at the possibilities for linkage mapping as well as the availability of reference sequences; 

+ Linkage mapping can be broken into three steps – linkage analysis, marker clustering and marker ordering.
+ Physical mapping

(3) quantitative trait analysis and genomic selection, including tools that perform quantitative trait locus (QTL) analysis in bi-parental populations, genome-wide association analysis (GWAS) and genomic selection and prediction. 

+ QTL analysis



Polyploid inheritance and simulation

+ mode of inheritance - 
  + disomic inheritance
  + Polysomic inheritance
  + intermediate inheritance (segmental allopolyploidy [3]) (mixosomy [4])

>  Currently there are no dedicated tools available to ascertain the most likely mode of inheritance in polyploids. 

+ Simulation software
  + PedigreeSim [5]
  + Polylink [6] (no longer available)
  + polySegratio [7]
  + HaploSim [8]



Quantitative trait loci (QTL)

> However, if a QTL has been fine mapped with respect to closely linked markers that are in linkage disequilibrium (LD) with the QTL, the associations between specific marker haplotypes and QTL alleles should hold across populations and need not be re-established for each individual family. [9]



"cM" = centimorgan



【Train note】

env: 

+ anaconda/3.6;
+ tensorflow_2.2
+ openmpi3

+ cuda/10.0.130

```bash
usage: model_by_keras.py [-h] -p PATH -1 TRAIN -2 VALID -o OUTPUT [-s SAMPLE]
                         [-a REGION] [-r ROUND] [-opt OPTIMIZER] [-plot PLOT]
                         [-sli SILENCE] [-loss LOSS]

optional arguments:
  -h, --help            show this help message and exit

Required:
  -p PATH, --path PATH  Input path for dataset.
  -1 TRAIN, --train TRAIN
                        Input train year.
  -2 VALID, --valid VALID
                        Input valid year.
  -o OUTPUT, --output OUTPUT
                        Input output dir.
  -s SAMPLE, --sample SAMPLE
                        number of sample e.g. 2000 for 2015_TCHBlup_2000.csv
  -a REGION, --region REGION
                        add regions (T/F)
  -r ROUND, --round ROUND
                        training round for esitimate average performance.
  -opt OPTIMIZER, --optimizer OPTIMIZER
                        select optimizer: Adam, SGD, rmsprop
  -plot PLOT, --plot PLOT
                        give plot? Output training plot for each training
  -sli SILENCE, --silence SILENCE
                        silent mode
  -loss LOSS, --loss LOSS
                        The target loss

```

geno data pre-process:

```python
import pandas as
data = pd.read_csv("../genos_SelTools.txt",sep=" ")
datat= data.T
data.drop(data.columns[0],inplace=True,axis=1)
datat.fillna("miss",inplace=True)
datat.index = datat.index.set_names("sample")
datat = datat.reset_index()
datat.to_csv("../fitted_genos.csv",sep="\t")
```







Prediction example code trunk:

```bash
srun -N 1 --mem=100G -p gpu --gres=gpu:tesla:1 --pty bash
module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /opt/ohpc/pub/apps/tensorflow_2.2
module load anaconda/3.6

# using RMSprop as optimizer, train set - 2016, valid set - 2017, 10 round for getting average accuracy
python ~/model_by_keras.py -p sugarcane_data/ -1 2016 -2 2017 -o models/ -r 10 #epochs 50 per opt
```

The whole test will introduce 3 optimizers - Adam, RMSprop and SGD;

The output will be a folder named [Train_year]\_vs\_[Valid_Year], contain the raw accuracy record for each round; and the average accuracy for whole round per optimizer;

Currently the model details can only be recorded manually with a table-like format: 

```bash
Model: "sequential" lr=0.00001
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 8694, 64)          384
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 4347, 64)          0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 4345, 128)         24704
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 2172, 128)         0
_________________________________________________________________
dropout (Dropout)            (None, 2172, 128)         0
_________________________________________________________________
flatten (Flatten)            (None, 278016)            0
_________________________________________________________________
dense (Dense)                (None, 8)                 2224136
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 72
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 72
_________________________________________________________________
dropout_1 (Dropout)          (None, 8)                 0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 9
=================================================================
Total params: 2,249,377
Trainable params: 2,249,377
Non-trainable params: 0
_________________________________________________________________

```

The training set: 80% of 2016 data; 

test set - 10% of 2016 data; 

In-same-year valid set - 10% of 2016 data; 

Future-valid-set 100% 2017 data;

AA = 2; AT, TA =1,TT=0, Miss = 0.01

in working: OneHotEncoding function

​	3 binary channels for AA, AT ,TT (0/1) 

​		markers  1    2    3

​		AA            0	0	1

​		AT/TA       1	0	0

​		TT             0	1	0

​	4 binary channels for regions

​	[21/OCT update] An alternative encoding: 3 channels for 4 categories:

​	markers  1    2    3

​	AA            0	0	1

​	AT/TA       1	0	0

​	TT             0	1	0       Set Missed data as 0 in all 3 channels.

```bash
python ~/GS_model.py -p sugarcane_data/ -1 2013-2015 -2 2017 -o models/ -r 10
```



Training set: 2013-2015

Valid set: 2016、2017



Sugarcane data 26k marker - marker A in chr1

10k marker - marker A in chr1 - Done





![train_den.png](https://github.com/CCS-voidBird/PhD_Notebook/blob/main/pic/2016-2017/train_den.png?raw=true)



![train_box.png](https://github.com/CCS-voidBird/PhD_Notebook/blob/main/pic/2016-2017/train_box.png?raw=true)

**Genetic linkage**



Sugarcane TCHBlup - AI vs GBLUP  -- ask Seema to get split methods (No major change)







Genomic selection

![img](https://ars.els-cdn.com/content/image/1-s2.0-S0378111920304649-gr4.jpg)

Question? the process of GS: 

	1. set a framework of the genotype markers array - sugarcane markers and select traits.
	2. training a genetic prediction model for the given markers ↑
	3. Using another strategy to generate natural offspring - F1 F2 .. from the original/natural/actual parents.
	4. Using the trained prediction model to predict phenotypes of these digital offspring -> select highest outcomes
	5. Record breeding traces for those outstanding offspring



upcoming paras：

number of filters (16, 32, 64, 128), regularization (i.e., weight decay in DL terminology, 0, 0.1, 0.01, 0.001), learning rate (0.1, 0.01, 0.001, 0.0025), number of neurons in fully connected layer (4, 8, 12, 16), number of hidden layers (1,5,10), and dropout (0, 0.01, 0.1, 0.2).





+ Linkage length(region)

> However, linkage analysis usually mapped the QTLs to a large interval of 20 centimorgans (cM) or more.

[Ref] Van Laere, A. S. et al. A regulatory mutation in *IGF2* causes a major QTL effect on muscle growth in the pig. *Nature* **425**, 832–836 (2003).

[Ref] Georges, M. et al. Mapping quantitative trait loci controlling milk production in dairy cattle by exploiting progeny testing. *Genetics* **139**, 907–920 (1995).



+ (Linkage disequilibrium) LD and the effect of effective population size.

> LD decays while marker distance increasing. By contrast, LD mapping relies on chromosome segments inherited from a common ancestor before the recorded pedigree — this is because it is the inheritance of identical chromosome segments by multiple descendents from a common ancestor that causes LD.



+ MAS vs GS

> The key difference between the two approaches is that MAS concentrates on a small number of QTLs that are tagged by markers with well-verified associations, whereas genomic selection uses a genome-wide panel of dense markers so that all QTLs are in LD with at least one marker. 



Kinds of MAS



principle of GS:

+ a sample of animals that have been assayed for the markers and recorded for the trait - reference population
+ Predict the genomic breeding value 





[1] Ng SB, Turner EH, Robertson PD, Flygare SD, Bigham AW, Lee C, Shaffer T, Wong M, Bhattacharjee A, Eichler EE, Bamshad M, Nickerson DA, Shendure J. Targeted capture and massively parallel sequencing of 12 human exomes. Nature. 2009 Sep 10;461(7261):272-6. doi: 10.1038/nature08250. Epub 2009 Aug 16. PMID: 19684571; PMCID: PMC2844771.

[2] Mamanova, Lira, et al. "Target-enrichment strategies for next-generation sequencing." *Nature methods* 7.2 (2010): 111-118.

[3] Stebbins, G. L. (1947). Types of polyploids: their classification and significance. *Adv. Genet.* 1, 403–429.

[4] Soltis, D. E., Visger, C. J., Marchant, D. B., and Soltis, P. S. (2016). Polyploidy: pitfalls and paths to a paradigm. *Am. J. Bot.* 103, 1146–1166. doi: 10.3732/ajb.1500501

[5] Voorrips, R. E., and Maliepaard, C. A. (2012). The simulation of meiosis in diploid and tetraploid organisms using various genetic models. *BMC Bioinformatics* 13:248. doi: 10.1186/1471-2105-13-248

[6] He, Y., Xu, X., Tobutt, K. R., and Ridout, M. S. (2001). Polylink: to support two-point linkage analysis in autotetraploids. *Bioinformatics* 17, 740–741. doi: 10.1093/bioinformatics/17.8.740

[7] Baker, P. (2014). polySegratio: simulate and test marker dosage for dominant markers in autopolyploids. R Package Version 0.2–4. doi: 10.1007/s00122-010-1283-z

[8] Motazedi, E., Finkers, R., Maliepaard, C., and De Ridder, D. (2017). Exploiting next-generation sequencing to solve the haplotyping puzzle in polyploids: a simulation study. *Br. Bioinformat.* doi: 10.1093/bib/bbw126 [Epub ahead of print].

[9] Khatkar, Mehar S., et al. "Quantitative trait loci mapping in dairy cattle: review and meta-analysis." *Genetics Selection Evolution* 36.2 (2004): 163-190.
