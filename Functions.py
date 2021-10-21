import pandas as pd

def read_pipes(genotype, phenotypes, years):
    selected_phenotypes = phenotypes\
        .pipe(lambda x: x.loc[x.Year in years])  #phenotypes.icol[phenotypes["Year"] in years]

    selected =