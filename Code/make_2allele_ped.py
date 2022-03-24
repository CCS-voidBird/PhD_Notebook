#make ped file with 2 alleles per SNP
import afxres
import pandas as pd
file_path = "E:/learning resource/PhD/genomic data/Sugarcane/genotypes.csv"
raw_data = pd.read_csv(file_path, sep='\t')

#Remove and save the sample column
sample_col = raw_data.pop('Sample')
print(raw_data.shape)
# make each row as listlenm
raw_data.replace(to_replace='--', value='00', inplace=True)
df_list = raw_data.values.tolist()
for x in range(len(df_list)):
    df_list[x] = [" ".join(list(i)) for i in df_list[x]]
#df_list = [" ".join(list("".join(x).replace("-","0"))) for x in df_list]

# convert to dataframe
df = pd.DataFrame(df_list)

print(df.shape)
#add the sample column back to the first column, rename first column to 'Clone'
df.insert(0, 'Clone', sample_col)
#df.rename(columns={df.columns[0]: 'Clone'}, inplace=True)

phenos = pd.read_csv("./phenotypes.csv", sep='\t')
phenos = phenos.query('Series in ["2013","2014","2015"]')

length = phenos.shape[0]

ped = pd.DataFrame(columns=['Clone','Clone1',"x1","x2","x3","TCHBlup"])
ped.Clone = phenos.Clone.tolist()
ped.Clone1 = range(1,length+1)
ped.x1 = [0]*length
ped.x2 = [0]*length
ped.x3 = [0]*length
ped.TCHBlup = phenos.TCHBlup.tolist()

ped = ped.merge(df, left_on='Clone', right_on='Clone',how='left')
ped.Clone = range(1,length+1)

ped.to_csv("./GCTAs/sugarcane_tch/sugarcane_tch.ped", sep='\t', index=False, header=False, na_rep='NA')
#save the dataframe to a new tab csv file
df.to_csv('E:/learning resource/PhD/genomic data/Sugarcane/genotypes_2allele.csv', sep='\t', index=False)
print(df[1:10,1:10])