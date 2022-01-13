import pandas as pd
import numpy as np
from Functions import decoding

file_path = "E:/learning resource/PhD/genomic data/Sugarcane/genotypes.csv"

genos = pd.read_csv(file_path,sep="\t")

genos2 = genos[(genos != "miss").all(axis=1)]

print(genos2.shape)

new_genos = genos2 #decoding(genos2)

new_genos.to_csv("E:/learning resource/PhD/genomic data/Sugarcane/qc_genotypes.csv",sep="\t",index=False)