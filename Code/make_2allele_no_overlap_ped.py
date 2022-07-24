#make ped file with 2 alleles per SNP
import afxres
import pandas as pd
file_path = "E:/learning resource/PhD/genomic data/Sugarcane/qc_genotypes_decode.csv"
print("Reading genotype file..")
raw_data = pd.read_csv(file_path, sep='\t')
#print(df.head(2))
#Remove and save the sample column

sample_col = raw_data.pop('Sample')
print(raw_data.shape)
# make each row as listlenm
raw_data.replace(to_replace=0.01, value='0 0', inplace=True)
raw_data.replace(to_replace=0, value='T T', inplace=True)
raw_data.replace(to_replace=1, value='A T', inplace=True)
raw_data.replace(to_replace=2, value='A A', inplace=True)

# convert to dataframe
df = pd.DataFrame(raw_data)

print(df.shape)
#add the sample column back to the first column, rename first column to 'Clone'
df.insert(0, 'Clone', sample_col)
#df.rename(columns={df.columns[0]: 'Clone'}, inplace=True)

phenos = pd.read_csv("E:/learning resource/PhD/genomic data/Sugarcane/phenotypes.csv", sep='\t')
blues = pd.read_csv("E:/learning resource/PhD/genomic data/Sugarcane/Blue_phenotypes.txt", sep='\t')
train_clones = phenos.query('Series in ["2013","2014","2015","2016"]').Clone.unique()
valid_clones = phenos.query('Series in ["2017"]').Clone.unique()
grm_clones = phenos.query('Series in ["2013","2014","2015","2016","2017"]').Clone.unique()
print(len(train_clones))

for i in range(3):
    dataset = [train_clones,valid_clones,grm_clones][i]
    ped = pd.DataFrame(columns=['Clone','Clone1',"x1","x2","x3","TCHBlup"])
    length = dataset.shape[0]
    ped.Clone = dataset
    ped.Clone1 = dataset
    ped.x1 = [0]*length
    ped.x2 = [0]*length
    ped.x3 = [0]*length
    ped.TCHBlup = blues.query("Clone in @dataset").TCHBlup.tolist()
    ped = ped.merge(df, left_on='Clone', right_on='Clone',how='left')
    set_group = ["train","valid","all"][i]
    ped.to_csv("E:/learning resource/PhD/genomic data/Sugarcane/GCTA/sugarcane_tch_{}.ped".format(set_group), sep='\t',
               index=False, header=False, na_rep='NA')
    print(ped.iloc[1:10,1:10])

#save the dataframe to a new tab csv file
#df.to_csv('E:/learning resource/PhD/genomic data/Sugarcane/genotypes_2allele.csv', sep='\t', index=False)
print(df[1:10,1:10])