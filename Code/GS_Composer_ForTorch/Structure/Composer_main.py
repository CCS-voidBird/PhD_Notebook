import os
from Structure.ModelProcessor import Model
from genotypeProcessor.data_loader import DataProcessor
from Structure.Visualization import TrainingHistoryPlotter
import pandas as pd

import numpy as np
from sklearn.metrics import mean_squared_error
import gc


class ML_composer:

    def __init__(self,args=None):

        self.args = args
        self._raw_data = {"GENO":pd.DataFrame(),"PHENO":pd.DataFrame(),"INDEX":pd.DataFrame(),"ANNOTATION":pd.DataFrame()}
        self._modelProcesser = Model(args)
        self.record = pd.DataFrame(columns=["Trait", "TrainSet", "ValidSet", "Model", "Test_Accuracy",
                          "Valid_Accuracy", "MSE", "loss" "Runtime", "Round"])
        
        self._info = {}
        self.data_provider = DataProcessor(args)
        self.plot_method = TrainingHistoryPlotter()
        self._info["CROSS_VALIDATE"] = self.data_provider.get_cross_validate()

    def prepare_cross_validate(self):
        ###Cross_validation;
        ###if use binary validate, index 0 will be validate set, and anyother index will be training set
        index_ref = []
        if self._info["CROSS_VALIDATE"] is None or self.args.vindex == 0:
            print("No cross validation index found, using whole dataset for training. \n")
            train_index = [0]
            return [(train_index,train_index)]
                    
        for idx in self._info["CROSS_VALIDATE"]:
            train_index = [x for x in self._info["CROSS_VALIDATE"] if x is not idx]
            valid_index = [idx]
            index_ref.append((train_index,valid_index))
            if self.args.vindex and idx == self.args.vindex:
                print("Detected manual validation index")
                print(f"Binary validation; index {idx} will be validate set, and anyother indexes will be training set \n")
                return [(train_index,valid_index)]

        return index_ref

    def compose(self,train_index:list,valid_index:list,val=1):

        self._modelProcesser.init_model()
        train_dataset, valid_dataset = self.data_provider.prepare_training(train_index,valid_index)


        print("Train status:")
        print("Epochs: ",self.args.epoch)
        print("Repeat(Round): ",self.args.round)

        round = 1
        
        while round <= self.args.round:
            print("Round: ",round)
            gc.collect()

            self._modelProcesser.init_model()
            history, runtime = self._modelProcesser.train(train_dataset,valid_dataset,round)

            """
            if self.args.plot is True:
                self.plot_method.plot(history, self.args.trait, self.args.output, round)
            
            if self.args.analysis is True:
                print("Start analysis model...")
                model_path = os.path.abspath(self.args.output) + "/{}_{}_{}".format(self._model["INIT_MODEL"].trait_label, self.model_name,val)
                if not os.path.exists(model_path) is True:
                    os.mkdir(model_path)
                investigate_model(model = self._model["TRAINED_MODEL"],
                                model_path=model_path,
                                ploidy=self.args.ploidy,marker_maf = np.array(self._info["MAF"]),args=self.args)
            """
            
            _, test_accuracy,_,_ = self.model_validation(train_index)
            mse, valid_accuracy, P, r2 = self._modelProcesser.validation(valid_dataset)

            #Convert time.time() to seconds and then minutes
            runtime = runtime.total_seconds() / 60

            for i in range(self.args.NumTrait):
                print("Trait ",i)
                print("Train accuracy: ", test_accuracy[i])
                print("Valid accuracy: ", valid_accuracy[i])
                print("MSE: ", mse[i])
                print("Special loss: ", r2[i])
                print("Runtime: ", runtime, " min")
                single_trait_record = [self.args.trait[i], train_index, valid_index, self._modelProcesser.model_name,
                                        test_accuracy[i], valid_accuracy[i], mse[i], r2[i], runtime.seconds, round]
                # add new record to the record dataframe
                self.record.loc[len(self.record)] = single_trait_record #[self.args.trait, train_index, valid_index, self.model_name,
                                #test_accuracy, valid_accuracy, mse, special_loss, runtime.seconds / 60, round]

            if self.args.predict is True:
                # predict the entire dataset
                print("Predicting the entire dataset...")
                entire_dataset = self.data_provider.make_dataset()

                prediction = self._modelProcesser._init_model.model(entire_dataset)


                pred_df = pd.DataFrame()
                pred_df["Sample"] = self.data_provider._raw_data["INDEX"].iloc[:, 1]
                pred_df["Index"] = self.data_provider._raw_data["INDEX"].iloc[:, -1]
                for i in range(self.args.NumTrait):
                    pred_df["Observed_{}".format(i)] = self.data_provider._raw_data["PHENO"].iloc[:,i] if self.args.NumTrait > 1 else self.data_provider._raw_data["PHENO"]
                    pred_df["Predicted_{}".format(i)] = prediction[:,i] if self.args.NumTrait > 1 else prediction
                if self.args.NumTrait == 1:
                    pred_df.to_csv(os.path.abspath(self.args.output) + "/{}_{}_{}_prediction.csv".format(self.args.trait, self.model_name, val), sep="\t", index=False)
                elif self.args.NumTrait > 1:
                    pred_df.to_csv(os.path.abspath(self.args.output) + "/MT_{}_{}_prediction.csv".format(self.model_name, val), sep="\t", index=False)

            if self.save == True:
                self.export_record()
            round += 1
            
        return

    def export_record(self):

        # sort record by traits
        self.record = self.record.sort_values(by=["Trait"])
        self.record.to_csv("{}/{}_train_record_{}.csv".format(os.path.abspath(self.args.output), self.model_name,self._model["INIT_MODEL"].trait_label), sep="\t", index=False)
        print("Result:")
        print(self.record)

        return


def main():
    print("Test mode:")



    

if __name__ == "__main__":
    main()




