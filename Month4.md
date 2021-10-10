Month 4 Notebook
====

Date: 01/Oct/2021 - 31/Oct/2021

<br> Editor: Chensong Chen
----

Current Goals:

+ Preparing training matrix
+ Write a proposal 

**continue: https://pubmed.ncbi.nlm.nih.gov/18076469/**

Add to Read list:

+ https://github.com/ne1s0n/coding_excercises

Li, S., Yang, G., Yang, S. *et al.* The development of a high-density genetic map significantly improves the quality of reference genome assemblies for rose. *Sci Rep* **9,** 5985 (2019). https://doi.org/10.1038/s41598-019-42428-y

[Done] Bourke PM, Voorrips RE, Visser RGF, Maliepaard C. Tools for Genetic Studies in Experimental Populations of Polyploids. Front Plant Sci. 2018 Apr 18;9:513. doi: 10.3389/fpls.2018.00513. PMID: 29720992; PMCID: PMC5915555.

Biemans F, de Jong MCM, Bijma P. Genetic parameters and genomic breeding values for digital dermatitis in Holstein Friesian dairy cattle: host susceptibility, infectivity and the basic reproduction ratio. Genet Sel Evol. 2019 Nov 20;51(1):67. doi: 10.1186/s12711-019-0505-3. PMID: 31747869; PMCID: PMC6865030.

Santos DJA, Cole JB, Lawlor TJ Jr, VanRaden PM, Tonhati H, Ma L. Variance of gametic diversity and its application in selection programs. J Dairy Sci. 2019 Jun;102(6):5279-5294. doi: 10.3168/jds.2018-15971. Epub 2019 Apr 10. PMID: 30981488.

T H E Meuwissen, B J Hayes, M E Goddard, Prediction of Total Genetic Value Using Genome-Wide Dense Marker Maps, *Genetics*, Volume 157, Issue 4, 1 April 2001, Pages 1819–1829, https://doi.org/10.1093/genetics/157.4.1819

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
+ 



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
+ 



```bash
srun -N 1 --mem=100G -p gpu --gres=gpu:tesla:1 --pty bash
module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /opt/ohpc/pub/apps/tensorflow_2.2
module load anaconda/3.6
source ~/.
# using RMSprop as optimizer
python ~/model_by_keras.py -p sugarcane_data/ -1 2016 -2 2017 -o models/ -r 10
```







**Genetic linkage**



paras：

number of filters (16, 32, 64, 128), regularization (i.e., weight decay in DL terminology, 0, 0.1, 0.01, 0.001), learning rate (0.1, 0.01, 0.001, 0.0025), number of neurons in fully connected layer (4, 8, 12, 16), number of hidden layers (1,5,10), and dropout (0, 0.01, 0.1, 0.2).



[1] Ng SB, Turner EH, Robertson PD, Flygare SD, Bigham AW, Lee C, Shaffer T, Wong M, Bhattacharjee A, Eichler EE, Bamshad M, Nickerson DA, Shendure J. Targeted capture and massively parallel sequencing of 12 human exomes. Nature. 2009 Sep 10;461(7261):272-6. doi: 10.1038/nature08250. Epub 2009 Aug 16. PMID: 19684571; PMCID: PMC2844771.

[2] Mamanova, Lira, et al. "Target-enrichment strategies for next-generation sequencing." *Nature methods* 7.2 (2010): 111-118.

[3] Stebbins, G. L. (1947). Types of polyploids: their classification and significance. *Adv. Genet.* 1, 403–429.

[4] Soltis, D. E., Visger, C. J., Marchant, D. B., and Soltis, P. S. (2016). Polyploidy: pitfalls and paths to a paradigm. *Am. J. Bot.* 103, 1146–1166. doi: 10.3732/ajb.1500501

[5] Voorrips, R. E., and Maliepaard, C. A. (2012). The simulation of meiosis in diploid and tetraploid organisms using various genetic models. *BMC Bioinformatics* 13:248. doi: 10.1186/1471-2105-13-248

[6] He, Y., Xu, X., Tobutt, K. R., and Ridout, M. S. (2001). Polylink: to support two-point linkage analysis in autotetraploids. *Bioinformatics* 17, 740–741. doi: 10.1093/bioinformatics/17.8.740

[7] Baker, P. (2014). polySegratio: simulate and test marker dosage for dominant markers in autopolyploids. R Package Version 0.2–4. doi: 10.1007/s00122-010-1283-z

[8] Motazedi, E., Finkers, R., Maliepaard, C., and De Ridder, D. (2017). Exploiting next-generation sequencing to solve the haplotyping puzzle in polyploids: a simulation study. *Br. Bioinformat.* doi: 10.1093/bib/bbw126 [Epub ahead of print].

[9] Khatkar, Mehar S., et al. "Quantitative trait loci mapping in dairy cattle: review and meta-analysis." *Genetics Selection Evolution* 36.2 (2004): 163-190.
