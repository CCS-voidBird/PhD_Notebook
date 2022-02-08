import pandas as pd
import configparser

"""
This script is used to transform the dataframe to a new dataframe with the table format in paper
"""



config = configparser.ConfigParser()
config.read("./DS_config.ini")
traits = config["BASIC"]["traits"].split("#")
print(traits)
record_path = "../../HPC_Results/MLP/MLP_train_record_by_default_summary.csv"
record_path = "../../HPC_Results/CNN/CNN_train_record_by_default_summary.csv"
df = pd.read_csv(record_path,sep="\t")
colnames = df.columns.tolist()
records = []
for group in ["in_year_accuracy","predict_accuracy"]:
    for units in [5,10,15,25]:
        record = []
        for trait in traits:
            for region in ["all","N","S","A","C"]:
                record.append(float('%.3f' % df[(df["trait"]==trait) & (df["n_units"]==units) & (df["Region"] == "all")][group].values[0]))
        print(record)
        records.append(record)
print(records)
df_records = pd.DataFrame(records)
csv_path = "../../HPC_Results/CNN/CNN_train_record_by_default_summary_table.csv"
df_records.to_csv(csv_path,sep="\t",index=False)