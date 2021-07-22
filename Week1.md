Week1 Notebook
====

Date: 19/July/2021 - 25/July/2021
<br> Editor: Chensong Chen
----

Current Goals:
1. Rerunning Kira's SimulationGenome software.
2. Literature review
    + Montesinos-López OA, Montesinos-López A, Pérez-Rodríguez P, Barrón-López JA, Martini JWR, Fajardo-Flores SB, Gaytan-Lugo LS, Santana-Mancilla PC, Crossa J. A review of deep learning applications for genomic selection. BMC Genomics. 2021 Jan 6;22(1):19. doi: 10.1186/s12864-020-07319-x. 
    + Abdollahi-Arpanahi R, Gianola D, Peñagaricano F. Deep learning versus parametric and ensemble methods for genomic prediction of complex phenotypes. Genet Sel Evol. 2020 Feb 24;52(1):12. doi: 10.1186/s12711-020-00531-z
    + Kemper KE, Bowman PJ, Pryce JE, Hayes BJ, Goddard ME. Long-term selection strategies for complex traits using high-density genetic markers. J Dairy Sci. 2012 Aug;95(8):4646-56. doi: 10.3168/jds.2011-5289. 
    + Bijma, P., Wientjes, Y.C.J., Calus, M.P.L.  Breeding top genotypes and accelerating response to recurrent selection by selecting parents with greater gametic variance (2020) Genetics, 214 (1), pp. 91-107. DOI: 10.1534/genetics.119.302643
    + Hickey, L.T., N. Hafeez, A., Robinson, H., Jackson, S.A., Leal-Bertioli, S.C.M., Tester, M., Gao, C., Godwin, I.D., Hayes, B.J., Wulff, B.B.H. Breeding crops to feed 10 billion (2019) Nature Biotechnology, 37 (7), pp. 744-754. Cited 133 times. DOI: 10.1038/s41587-019-0152-9
    + Dias, R., Torkamani, A. Artificial intelligence in clinical and genomic diagnostics. Genome Med 11, 70 (2019). https://doi.org/10.1186/s13073-019-0689-8

    
Notes:
1. Descriptions from Kira's message:
> GA inputs:  
>+ params.txt (Parameters: used to tell the GA the size of the segment_effects.txt matrix, how many lines it should select, how many generations to run for, etc. The R script I sent you modifies this before running the GA.)
>+ segment_effects.txt (rows = haplotypes of simulated lines, columns = local GEBVs for a particular block. The R script I sent you creates this.)
>+ iseed.txt (you can change the random number generator seed using this. I don't usually bother.)

> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.

2. Notes from literatures:
    1.Today, genomic selection (GS), proposed by Bernardo [[1]](#"1") and Meuwissen et al. [^2] has become an established methodology in breeding.





> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.
> GA output:
>+ Creates the file best.chr (I don't know how to interpret most of this file, but we read the GA's final choice from the last row, excluding the first two columns. The numbers in that final row are the 1-based index/row number of the genotypes in segment_effects.txt. The R script I sent you reads the selection results from this file.)
>+ Creates the file best.fitness (1st column is each generation number when it found a set with a higher fitness score. That fitness score is in column 3.)
>+ Prints a log, which we save as ga.log (I believe that: column 1 is GA generation number. Column 2 is average score of the sets of genotypes in this generation. Column 3 is score of the best set of genotypes in this generation. Column 4 is the score of the best set of genotypes it has found so far.) All scores in this file are rounded to whole numbers, so best.fitness gives more detail.



























Reference:

<span id="1">[1]</span> Bernardo, R. (1994), Prediction of Maize Single-Cross Performance Using RFLPs and Information from Related Hybrids. Crop Science, 34: 20-25 cropsci1994.0011183X003400010003x. https://doi.org/10.2135/cropsci1994.0011183X003400010003x
[^2] Meuwissen TH, Hayes BJ, Goddard ME. Prediction of total genetic value using genome-wide dense marker maps. Genetics. 2001 Apr;157(4):1819-29. PMID: 11290733; PMCID: PMC1461589.