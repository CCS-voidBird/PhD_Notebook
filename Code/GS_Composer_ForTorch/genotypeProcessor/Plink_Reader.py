import pandas as pd
import numpy as np


def read_transform_plink_files(geno_path):

    print("Reading geno files from: ",geno_path)
    """
    convert plink format to pandas dataframe
    """
    full_ped_name = geno_path + ".ped"
    full_map_name = geno_path + ".map"
    #full_fam_name = geno_path + ".fam"

    map_data = pd.read_csv(full_map_name,sep=r"\s+",header=None)
    print("Read {} markers from map file.".format(map_data.shape[0]))
    #check map file infos, if V3 == V4, then raise a warning, then let V4 = 0
    if map_data.iloc[:,2].equals(map_data.iloc[:,3]):
        #print("V3 is same with V4, set V4 to 0")
        map_data.iloc[:,3] = 0

    marker_list = map_data.iloc[:,1].values
    #sample_list = fam_data.iloc[:,1].values
    #ref_allele = map_data.iloc[:,3].values
    

    ped = pd.read_csv(full_ped_name,sep=r"\s+",header=None)
    
    #ped = pd.read_csv(full_ped_name,sep="\t",header=None)
    ped_1_field = ped.iloc[:,0:6]
    ped_2_field = ped.iloc[:,6:] 
    ped_2_field.replace(["--",-9,"NA"],np.nan,inplace=True)
    #Identify minor allele with lower allele frequyency
    print("Sorting minor allele.......", end="")
    ped_2_field_minor_allele = ped_2_field.apply(lambda x: x.value_counts().idxmin(),axis=0)
    ped_2_field_minor_allele.replace("0","T",inplace=True)
    ##Reindex the minor allele from 0 to the last marker
    ped_2_field_minor_allele.index = ped_2_field_minor_allele.index-6
    print("DONE")#,ped_2_field_minor_allele.head())
    #ped_2_field_minor_allele = ped_2_field_minor_allele.replace("0","T")

    if ped_2_field.shape[1] == len(marker_list):
        print("The number of markers are same with map file presented.")
        ped_2_field.columns = marker_list
        #check if values in variant_allele is any integer or not
        #print(ped_2_field.dtypes)
        if ped_2_field.dtypes.unique().all() == "int64":
            print("Variant allele is integer, no need to transform.")
        else:
            # count maximum string length across entire dataframe
            ped_2_field.replace(" ","",inplace=True)
            ploidy = ped_2_field.values.astype(str).max(axis=0).max()
            print("Detected ploidy: ",ploidy)

            # for each individual x and each marker i, count the refernence allele number from the ped_2_field_minor_allele[i]
            for i, col in enumerate(ped_2_field.columns):
                target_char = ped_2_field_minor_allele[i]
                ped_2_field[col] = ped_2_field[col].apply(lambda x: ploidy - x.count(target_char))

    elif ped_2_field.shape[1] % len(marker_list) == 0 and ped_2_field.shape[1] > len(marker_list):
        ploidy = ped_2_field.shape[1] // len(marker_list)
        print("Detected ploidy: ",ploidy)
        
        #reshape the ped genos from (N,marker*ploidy) to (N,marker,ploidy)
        ped_2_field_reshaped = np.reshape(ped_2_field.values,(ped_2_field.shape[0],len(marker_list),ploidy))

        if ped_2_field.dtypes.unique().all() == "int64":
            print("Variant allele is integer, no need to transform.")
        else:
            
            print("Variant allele is string, transforming...Now convert allele genotypes to allele counts.")
            #print(ped_2_field_reshaped[0,0,:])
            for i in range(len(marker_list)):
                ped_2_field_reshaped[:,i,:] = np.where(ped_2_field_reshaped[:,i,:] == ped_2_field_minor_allele[i], 0, 1)

        #sum the ploidy to get the final geno data
        ped_2_field = np.sum(ped_2_field_reshaped,axis=2)

    else:
        print("Cannot match the marker number with the genotype data.")
        exit()
    # combine the ped_1_field and ped_2_field
    ped_2_field_merged = pd.DataFrame(np.concatenate([ped_1_field,ped_2_field],axis=1))
    print("Finish transforming genotype data.")
    #print(ped_2_field_merged.head())
    return ped_2_field_merged, map_data
