import pandas as pd
import torch

trait_path = "E:/learning resource/PhD/phenotypes.csv"
genos_path = "E:/learning resource/PhD/geno_data1.csv"

phenotypes = pd.read_csv(trait_path,sep="\t")
genos = pd.read_csv(genos_path,sep="\t")
print(genos.shape)
print(phenotypes.shape)

merged_data = phenotypes
samples = phenotypes["Clone"]
print(samples)