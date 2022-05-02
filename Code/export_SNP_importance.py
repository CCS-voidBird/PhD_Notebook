from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
# import plt
import matplotlib.pyplot as plt

traits = ["TCHBlup","CCSBlup","FibreBlup"]
regions = ["all","A","N","S","C"]
importances = []
combos = []
for trait in traits:
    for region in regions:
        combos = combos + [(trait,region)]
        # Load the model
        model = joblib.load("E:/learning resource/PhD/HPC_Results/RF_region/rerun/models/"+trait+"_RF_"+region+"_model.json")
        # Get the feature importances as list with its trait and region
        importances.append([trait,region]+list(model.feature_importances_))
        print(model.feature_importances_.shape)

# Convert to dataframe
importances = pd.DataFrame(importances,columns=["Trait","Region"]+list(range(1,model.feature_importances_.shape[0]+1)))
print(importances.shape)
# write to csv by tab
importances.to_csv("E:/learning resource/PhD/HPC_Results/RF_region/rerun/RF_SNP_importances.csv",sep="\t",index=False)
