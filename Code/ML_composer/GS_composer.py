

from Functions import *
from ClassModel import *
from GS_interpretation import *
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from Composer_ArgsPaeser import get_args
import pandas as pd
import matplotlib.pyplot as plt
import keras.utils
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
import gc
import os

"""
import sklearn.preprocessing
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import platform
from sklearn.preprocessing import OrdinalEncoder
import configparser
"""


def plot_loss_history(h, title,plot_name=None,checkpoint=0):
    print("Plotting loss history...")
    hist_df = pd.DataFrame(h.history)
    hist_df['round'] = abs(checkpoint)
    hist_df['epoch'] = hist_df.index
    checkpoint = abs(checkpoint)
    try:
        history_record = pd.read_csv(plot_name+"_history.csv", sep="\t")
        history_record = history_record.append(hist_df)
        history_record.to_csv(plot_name+"_history.csv", sep="\t",index=False)
    except:
        hist_df.to_csv(plot_name+"_history.csv", sep="\t",index=False)


    plot_name_loss=plot_name+"_"+str(checkpoint)+"_loss.png"
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(h.history['loss'][1:], label = "Train loss", color = "blue")
    axs[0].plot(h.history['val_loss'][1:], label = "Validation loss", color = "red")
    axs[0].set_ylabel("MSE")

    axs[1].plot(h.history['p_corr'][1:], label = "Train cor", color = "blue")
    axs[1].plot(h.history['val_p_corr'][1:], label = "Validation cor", color = "red")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel("Pearson's Correlation")
    fig.suptitle(title)
    #print plot name
    print("Plot name: ", plot_name)
    if plot_name:
        plt.legend()
        plt.savefig(plot_name_loss)
        plt.close()

        #read history csv file from path
        

    else:
        return
    
    
def plot_correlation(predictions, observations, title,plot_name=None,checkpoint=0):

    ###create clear correlation plot for predictions and observations

    print("Plotting predictions and observations..")
    hist_df = pd.DataFrame()
    hist_df['Individual'] = [x for x in range(len(predictions))]
    hist_df['Prediction'] = predictions
    hist_df['Observation'] = observations
    hist_df.sort_values(by='Observation', inplace=True)
    try:
        history_record = pd.read_csv(plot_name+title+"_correlation.csv", sep="\t")
        history_record = history_record.append(hist_df)
        #history_record.to_csv(plot_name+title+"_correlation.csv", sep="\t",index=False)
    except:
        hist_df.to_csv(plot_name+"_correlation.csv", sep="\t",index=False)


    plot_name_loss=plot_name+"_"+str(checkpoint)+"_correlation.png"
    fig = plt.figure()
    fig, axs = plt.subplots(1, 1)
    axs.scatter(x=hist_df['Individual'],y=hist_df['Observation'], label = "Observation", color = "blue")
    axs.scatter(x=hist_df['Individual'],y=hist_df['Prediction'], label = "Prediction", color = "red")
    #axs[0].plot(hist_df['val_loss'][1:], label = "Validation loss", color = "red")
    axs.set_ylabel(" ")
    fig.suptitle(title)
    #print plot name
    print("Plot name: ", plot_name)
    if plot_name:
        plt.legend()
        plt.savefig(plot_name_loss)
        plt.close()
        

    else:
        return

    #plt.show()

def plot_corr_history(h, title,plot_name=None,checkpoint=0):
    
    print("Plotting correlation history...")
    plot_name_corr=plot_name+"_corr.png"
    corr_plot = plt.figure()
    corr_plot.plot(h.history['p_corr'][5:], label = "Train cor", color = "blue")
    corr_plot.plot(h.history['val_p_corr'][5:], label = "Validation cor", color = "red")
    corr_plot.xlabel('Epochs')
    corr_plot.title(title)
    #print plot name
    print("Plot name: ", plot_name)
    if plot_name and checkpoint == 0:
        #plt.legend()
        corr_plot.savefig(plot_name_corr)
        corr_plot.close()
    else:
        pass

lr_opt = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.005, cooldown=0, min_lr=0)

class ML_composer:

    def __init__(self,silence=0,args=None):
        self._raw_data = {"GENO":pd.DataFrame(),"PHENO":pd.DataFrame(),"INDEX":pd.DataFrame(),"ANNOTATION":pd.DataFrame()}
        self.train_data = None
        self.train_info = pd.DataFrame()
        self.train_pheno = pd.DataFrame()
        self.valid_data = None
        self.valid_info = pd.DataFrame()
        self.valid_pheno = pd.DataFrame()
        self._model = {"INIT_MODEL":Model(args=args),"TRAINED_MODEL":Model(args=args)}
        self._info = {}
        self.method = None
        self.modelling = None
        self.silence_mode = silence
        self.config = None
        self.save = True
        self.plot = False
        self.args = None
        self.batchSize = 16
        self.mean_pheno = 0
        self.subset_ratio = 1
        self.record = pd.DataFrame(columns=["Trait", "TrainSet", "ValidSet", "Model", "Test_Accuracy",
                          "Valid_Accuracy", "self.args.loss", "Runtime"])
        self.model_name = None

    def load_data(self,raw_data,raw_model,raw_info):
        #read outside dict of data
        self._raw_data = raw_data
        self._model = raw_model
        self._info = raw_info

        return

    def get_data(self,configer,args):
        self.args = args
        self.config = configer
        self.record = pd.DataFrame(columns=["Trait", "TrainSet", "ValidSet", "Model", "Test_Accuracy",
                          "Valid_Accuracy", "MSE", self.args.loss, "Runtime"])
        self._raw_data["GENO"],self._raw_data["MAP"] = read_transform_plink_files(self.args.geno)
        
        #self._raw_data["MAP"] = pd.read_table(args.geno + ".map", delim_whitespace=True,header=None)
        self._raw_data["FAM"] = pd.read_table(args.geno + ".fam", delim_whitespace=True,header=None)
        self._raw_data["PHENO"] = pd.read_table(args.pheno, delim_whitespace=True,header=None)
        self._raw_data["INDEX"] = pd.read_table(args.index,delim_whitespace=True,header=None)
        self._raw_data["ANNOTATION"] = pd.read_table(args.annotation,delim_whitespace=True) if args.annotation is not None else None
        self._info["CROSS_VALIDATE"] = sorted(self._raw_data["INDEX"].iloc[:,-1].unique())
        self._info["MARKER_SIZE"] = self._raw_data["MAP"].shape[0]
        self._info["MAF"] = self._raw_data["GENO"].iloc[:,6:].apply(lambda x: np.mean(x)/self.args.ploidy,axis=0)
        self.batchSize = args.batch
        #print(self._raw_data["INDEX"].iloc[:,-1].value_counts().sort_values())

        self._raw_data["INFO"] = self._raw_data["FAM"].iloc[:,0:6]  #Further using fam file instead.

        print("Get genotype shape:",self._raw_data["GENO"].iloc[:,6:].shape)
        #print(self._raw_data["GENO"].iloc[:,6:].iloc[1:10,1:10])
        self.plot = self.args.plot
        self.sort_data()
        if self.args.annotation is not None:
            annotation_groups = self._raw_data["ANNOTATION"].iloc[:, -1].unique()
            anno_dict = {annotation_groups[x]:x for x in range(len(annotation_groups))}
            self.annotation = self._raw_data["ANNOTATION"]
            self.annotation.iloc[:,-1] = self.annotation.iloc[:,-1].map(anno_dict)
            #print(self.annotation.head(10))
            if self.args.model == "MultiLevel Attention":
                LD_index = self.annotation.groupby('LID')
                self.annotation = np.array([x.values for x in LD_index.groups.values()])
                #print(self.annotation)
            else:
                self.annotation = to_categorical(np.asarray(self.annotation.iloc[:, 2]).astype(np.float32))
            # self.annotation = np.asarray(self.annotation.iloc[:, 2]).astype(np.float32)
                print("Got LD shape:")
                print(self.annotation.shape)

        return

    def sort_data(self):
        """
        Sort raw data as plink format
        FID,IID,father,mother,sex,pheno --> fam
        Chromosome, Variant ID, position, base pair --> map
        """
        # sort GENO by first col with reference FAM
        print("Running data check")
        #change all FID and IID into string
        self._raw_data["FAM"].iloc[:,0:2] = self._raw_data["FAM"].iloc[:,0:2].astype(str)
        self._raw_data["GENO"].iloc[:,0:2] = self._raw_data["GENO"].iloc[:,0:2].astype(str)
        self._raw_data["MAP"].iloc[:,0:2] = self._raw_data["MAP"].iloc[:,0:2].astype(str)
        self._raw_data["PHENO"].iloc[:,0:2] = self._raw_data["PHENO"].iloc[:,0:2].astype(str)
        self._raw_data["INDEX"].iloc[:,0:2] = self._raw_data["INDEX"].iloc[:,0:2].astype(str)
        sample_reference = self._raw_data["FAM"].iloc[:,1]## Get fam IID as reference
        modelling_reference = self._raw_data["INDEX"].iloc[:,1]
        snp_reference = self._raw_data["MAP"].iloc[:,:2]
        """
        for label in ["GENO","FAM","PHENO","INDEX"]:
            #check if samples are aligned with same order
            print(label)
           
            if self._raw_data[label].iloc[:,1].equals(sample_reference) is False:
                #check if samples are aligned with same order
                print("Samples are not aligned with same order or not the same name style, please double-check.")
                print("GS Composer would only use the samples recorded in the fam file.")
                
                print(sample_reference.head(10))
                print(self._raw_data[label].iloc[:,1].head(10))
                #exit()
        """
        
        if self._raw_data["GENO"].iloc[:,6:].shape[1] != snp_reference.shape[0]:
            print("SNPs are not in same length in ped file and map file")
            print("SNP length in ped file: ",self._raw_data["GENO"].iloc[:,6:].shape[1])
            print("SNP length in map file: ",snp_reference.shape[0])
            ##select ped records that their second column value are appeared in fam file

        ##sort ped, pheno and index file by IID order from fam file
        self._raw_data["GENO"] = self._raw_data["GENO"].loc[self._raw_data["GENO"].iloc[:,1].isin(modelling_reference),:].reset_index(drop=True)
        #print(self._raw_data["GENO"].iloc[1:10,1:10])
        self._raw_data["PHENO"] = self._raw_data["PHENO"].loc[self._raw_data["PHENO"].iloc[:,1].isin(modelling_reference),:].reset_index(drop=True)
        #print(self._raw_data["INDEX"].loc[self._raw_data["INDEX"].iloc[:,1].isin(sample_reference),:])
        self._raw_data["INDEX"] = self._raw_data["INDEX"].loc[self._raw_data["INDEX"].iloc[:,1].isin(modelling_reference),:].reset_index(drop=True)
        print("SNPs are filtered by fam file.")
        #Check if samples are aligned with same order with fam file


        if self.args.annotation is not None and self._raw_data["ANNOTATION"].iloc[:,:1].equals(snp_reference) is False:
            print("SNPs in annotation file are not ordered by map file")
            #exit()
        print("Data check & sort passed.")

        ## data quality control and filter by MAF
        if self.args.maf > 0:
            print("Filtering SNPs by MAF...")
            print("Loaded SNPs: ",self._info["MARKER_SIZE"])
            print("Filtering SNPs with MAF lower than {} with ployidy {}.".format(self.args.maf,self.args.ploidy))
            ### Get column index of MAF >= self.args.maf 
            selected_marker_idx = self._info["MAF"][self._info["MAF"] >= self.args.maf].index
            #print(selected_marker_idx[0:10])
            self._raw_data["GENO"] = self._raw_data["GENO"].iloc[:,selected_marker_idx]
            self._raw_data["MAP"] = self._raw_data["MAP"].iloc[selected_marker_idx-6,:] #update map file
            self._info["MARKER_SIZE"] = self._raw_data["MAP"].shape[0]
            print("Filtered SNPs: ",self._info["MARKER_SIZE"])


    def prepare_model(self):
        # create a Model object

        if self.args.load is not None:
            self._model["INIT_MODEL"].load_model(self.args.load)
        else:
            self._model["INIT_MODEL"].init_model()
        # save model summary to a txt file under the output directory
        self.model_name = self._model["INIT_MODEL"].get_model_name()

        # get data requirements - dimension, annotations, etc
        return

    def prepare_cross_validate(self):
        ###Cross_validation;
        ###if use binary validate, index 0 will be validate set, and anyother index will be training set
        index_ref = []
        for idx in self._info["CROSS_VALIDATE"]:
            train_index = [x for x in self._info["CROSS_VALIDATE"] if x is not idx]
            valid_index = [idx]
            index_ref.append((train_index,valid_index))
            if self.args.vindex != 0 and idx == self.args.vindex:
                print("Detected manual validation index")
                print(f"Binary validation; index {idx} will be validate set, and anyother indexes will be training set")
                return [(train_index,valid_index)]

        return index_ref

    def prepare_training(self,train_index:list,valid_index:list):

        removal = np.where(self._raw_data["PHENO"].iloc[:, int(self.args.mpheno)+1].isin([None,"NA",np.nan]))[0].tolist()
        print("Overall population: {}".format(len(self._raw_data["INDEX"].index)))
        print("{} individuals need to be removed due to the miss phenotype".format(len(removal)))
        train_mask = [x for x in np.where(self._raw_data["INDEX"].iloc[:, -1].isin(train_index))[0].tolist() if x not in removal]
        valid_mask = [x for x in np.where(self._raw_data["INDEX"].iloc[:, -1].isin(valid_index))[0].tolist() if x not in removal]
        print("Filtered population: {}".format(len(train_mask)+len(valid_mask)))
        #print(self.args.mafm)
        self.train_data = self._raw_data["GENO"].iloc[train_mask, 6:] * self._info["MAF"] if self.args.mafm is True else self._raw_data["GENO"].iloc[train_mask, 6:]
        self.valid_data = self._raw_data["GENO"].iloc[valid_mask, 6:] * self._info["MAF"] if self.args.mafm is True else self._raw_data["GENO"].iloc[valid_mask, 6:]

        self.train_pheno = self._raw_data["PHENO"].iloc[train_mask,self.args.mpheno + 1]
        print("Mean of train phenotype:",np.mean(self.train_pheno))
        self.mean_pheno = np.mean(self.train_pheno)
        if self.args.mean is not True or self.args.data_type == "ordinal":
            print("Use raw phenotype as the target")
            self.mean_pheno = 0
        self.train_pheno = self.train_pheno - self.mean_pheno
        self.valid_pheno = self._raw_data["PHENO"].iloc[valid_mask, self.args.mpheno + 1]
        #print(self._raw_data["PHENO"].iloc[valid_mask,])
        #self.valid_pheno = self.valid_pheno - self.mean_pheno
        #print(self.valid_pheno.head(5))
        self.train_pheno = np.asarray(self.train_pheno).astype(np.float32)
        self.valid_pheno = np.asarray(self.valid_pheno).astype(np.float32)
        if self.args.data_type == "ordinal":
            self.args.classes = np.max(self.train_pheno) + 1
            #encoder = OrdinalEncoder()
            try:
                self.args.classes = int(max(np.max(self.train_pheno),np.max(self.valid_pheno)))
                #self.train_pheno = keras.utils.to_ordinal(self.train_pheno)
                #self.valid_pheno = keras.utils.to_ordinal(self.valid_pheno)
            except:
                print("Using backup function inherited from keras source code. "
                      "You may need to use code 'pip install tf-nightly' to install the actual module.")
                self.args.classes = int(max(np.max(self.train_pheno), np.max(self.valid_pheno)))
                print(self.args.classes)
                #self.train_pheno = to_ordinal(self.train_pheno)
                #self.valid_pheno = to_ordinal(self.valid_pheno)
                #print(self.valid_pheno.shape)
        self._model = {"INIT_MODEL": Model(self.args), "TRAINED_MODEL": Model(self.args)}
        self.prepare_model()

        return

    def train(self,features_train, features_test, target_train, target_test,round=1):

        if type(features_train) is list:
            n_features = features_train[0].shape[1:]
        else:
            n_features = features_train.shape[1:]

        print("Got input shape:",n_features)
        self._model["TRAINED_MODEL"] = self._model["INIT_MODEL"].modelling(
            input_shape = n_features,args = self.args, lr=float(self.args.lr),annotation = self.annotation) if self.args.annotation else self._model["INIT_MODEL"].modelling(
            input_shape = n_features,args = self.args, lr=float(self.args.lr))
        if round == 1:
            with open(os.path.abspath(self.args.output) + "/model_summary.txt", "w") as fh:
                self._model["TRAINED_MODEL"].summary(print_fn=lambda x: fh.write(x + "\n"))
            try:
                keras.utils.vis_utils.plot_model(self._model["TRAINED_MODEL"],
                                                 to_file=os.path.abspath(self.args.output) + "/model_summary.png",
                                                 show_shapes=True, show_layer_names=True)
            except:
                "Model plotting function error"

        try:
            print(self._model["TRAINED_MODEL"].summary())
        except:
            print("It is a sklean-Random-forest model.")

        startTime = datetime.now()

        history = self._model["TRAINED_MODEL"].fit(
            features_train, target_train,
            epochs=int(self.args.epoch),
            validation_data=(features_test, target_test), verbose=int(self.args.quiet),
            batch_size = self.batchSize,callbacks=[lr_logger]) #callbacks=[self._model["INIT_MODEL"].lr_schedule],


        # let's just print the final loss
        print(' - train loss     : ' + str(history.history['loss'][-1]))
        print(' - validation loss: ' + str(history.history['val_loss'][-1]))
        print(' - loss decrease rate in last 5 epochs: ' + str(
            np.mean(np.gradient(history.history['val_loss'][-5:]))))
        print(' - Actual Training epochs: ', len(history.history['loss']))
        #print(self._model["TRAINED_MODEL"].predict(features_test).shape)
        test_length = target_train.shape[0]
        y_pred = self._model["TRAINED_MODEL"].predict(features_train,batch_size=self.batchSize,verbose=int(self.args.quiet))
        if self.args.data_type == "ordinal":
            y_pred = tf.reduce_sum(tf.round(y_pred),axis=-1)
            y_pred = np.reshape(y_pred, (test_length,))
            #print(y_pred.shape)
            #print(target_train.shape)
            #test = tf.reduce_sum(target_train,axis=-1)
            test_accuracy = np.corrcoef(y_pred, target_train)[0, 1]
        else:
            y_pred = np.reshape(y_pred, (test_length,))
            test_accuracy = np.corrcoef(y_pred, target_train)[0, 1]
        print("Train End.")
        print("In-year accuracy (measured as Pearson's correlation) is: ", test_accuracy)
        endTime = datetime.now()
        runtime = endTime - startTime
        print("Training Runtime: ", runtime.seconds / 60, " min")
        
        return history,test_accuracy,runtime

    def compose(self,train_index:list,valid_index:list,val=1):

        features_train,target_train = self._model["INIT_MODEL"].data_transform(self.train_data,self.train_pheno, pheno_standard = self.args.rank)
        features_val,target_val = self._model["INIT_MODEL"].data_transform(self.valid_data,self.valid_pheno, pheno_standard = self.args.rank)
        num_samples = len(features_train)
        indices = np.random.permutation(num_samples)
        features_train = features_train[indices]
        target_train = target_train[indices]
        print("Train status:")
        print("Epochs: ",self.args.epoch)
        print("Repeat(Round): ",self.args.round)
        #print("feature shape:",features_train.shape)

        round = 1
        val_record = -1
        while round <= self.args.round:
            self._model["TRAINED_MODEL"] = None
            keras.backend.clear_session()
            gc.collect()
            history, test_accuracy, runtime = self.train(features_train, features_val, target_train, target_val,round=round)
            valid_accuracy, mse,special_loss = self.model_validation()
            if valid_accuracy > val_record:
                val_record = valid_accuracy
                if self.args.save is True:
                    print("Saving the model with higher accuracy...")
                    try:
                        self._model["TRAINED_MODEL"].save(
                            os.path.abspath(self.args.output) + "/{}_{}_{}".format(self.args.trait, self.model_name,
                                                                                   val))
                        print("Model saved.")

                        self._model["TRAINED_MODEL"].save(
                            os.path.abspath(self.args.output) + "/{}_{}_{}.h5".format(self.args.trait, self.model_name,
                                                                                   val))
                        print("h5 Model saved.")
                    except:
                        print("Saving model failed, tring directly save by using self._model[\"TRAINED_MODEL\"].save")
                if self.args.analysis is True:
                    print("Start analysis model...")
                    model_path = os.path.abspath(self.args.output) + "/{}_{}_{}".format(self.args.trait, self.model_name,val)
                    if not os.path.exists(model_path) is True:
                        os.mkdir(model_path)
                    investigate_model(model = self._model["TRAINED_MODEL"],
                                      model_path=model_path,
                                      ploidy=self.args.ploidy,args=self.args)

            self.record.loc[len(self.record)] = [self.args.trait, train_index, valid_index, self.model_name,
                               test_accuracy, valid_accuracy, mse, special_loss, runtime.seconds / 60]
            check_usage()
            check_gpu_usage()
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                try:
                    mem_usage = tf.config.experimental.get_memory_usage('GPU:0')
                    print("Currently using GPU memory: {} GB".format(mem_usage/1e9))
                except:
                    print("Checking memory usage is not currently available.")

            if self.args.predict is True:
                # predict the entire dataset
                print("Predicting the entire dataset...")
                features_all, target_all = self._model["INIT_MODEL"].data_transform(
                    self._raw_data["GENO"].iloc[:, 6:], self._raw_data["PHENO"].iloc[:, self.args.mpheno + 1], pheno_standard=self.args.rank)
                y_pred_all = self._model["TRAINED_MODEL"].predict(features_all, batch_size=self.batchSize,verbose=int(self.args.quiet)) + self.mean_pheno
                if self.args.data_type == "ordinal":
                    y_pred_all = tf.reduce_sum(tf.round(y_pred_all), axis=-1)
                    y_pred_all = np.reshape(y_pred_all, (len(target_all),))
                    
                else:
                    y_pred_all = np.reshape(y_pred_all, (len(target_all),))
                print("Predicted: ", np.array(y_pred_all[:10]))
                print("Observed: ", np.array(target_all[:10]))
                # Save samples,validate index,obvserved and predicted values to a file
                print("Saving the prediction results...")
                pred_df = pd.DataFrame()
                pred_df["Sample"] = self._raw_data["INDEX"].iloc[:, 1]
                pred_df["Index"] = self._raw_data["INDEX"].iloc[:, -1]
                pred_df["Observed"] = target_all
                pred_df["Predicted"] = y_pred_all
                pred_df.to_csv(os.path.abspath(self.args.output) + "/{}_{}_{}_prediction.csv".format(self.args.trait, self.model_name, val), sep="\t", index=False)


            if self.plot is True:
                # create a folder to save the plot, folder name: trait, model
                print("Plotting the training process...")
                plot_dir = os.path.abspath(self.args.output) + "/{}_{}_{}".format(self.args.trait, self.model_name,
                                                                                  self.args.trait)
                print(plot_dir)
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                # create a file name for the plot: path, model name, trait and round
                plot_name = plot_dir + "/{}_{}_{}".format(self.args.trait, self.model_name, val)
                # plot_name = os.path.abspath(self.args.output) + "/" + self.args.model + "_" + self.args.trait + "_" + str(round) + ".png"
                plot_loss_history(history, self.args.trait, plot_name,round-self.args.round)

            if self.save == True:
                self.export_record()
            round += 1
            
        return

    def model_validation(self):

        valid_data,valid_pheno = self._model["INIT_MODEL"].data_transform(
            self.valid_data,self.valid_pheno, pheno_standard = self.args.rank)
        print("Predicting valid set..")
        val_length = valid_pheno.shape[0]

        y_pred_valid = self._model["TRAINED_MODEL"].predict(valid_data,batch_size=self.batchSize,verbose=int(self.args.quiet))+self.mean_pheno
        if self.args.data_type == "ordinal":
            y_pred_valid = tf.reduce_sum(tf.round(y_pred_valid),axis=-1)
            y_pred_valid = np.reshape(y_pred_valid, (val_length,))
            #print(y_pred_valid.shape)
            #print(valid_pheno.shape)
            #test = tf.reduce_sum(valid_data,axis=-1)
            accuracy_valid = np.corrcoef(y_pred_valid, valid_pheno)[0, 1]
        else:
            y_pred = np.reshape(y_pred_valid, (val_length,))
            accuracy_valid = np.corrcoef(y_pred, valid_pheno)[0, 1]

        mse = mean_squared_error(y_pred_valid, valid_pheno)
        print("Testing prediction:")
        print("Predicted: ", np.array(y_pred_valid[:10]))
        print("observed: ", np.array(valid_pheno[:10]))
        plot_correlation(y_pred_valid,valid_pheno,self.args.trait,
                         os.path.abspath(self.args.output) + "/{}_{}_{}".format(self.args.trait, self.model_name, self.args.trait))
        print("Observation mean: {} Var: {}".format(np.mean(valid_pheno), np.var(valid_pheno)))
        print("Prediction mean: {} Var: {}".format(np.mean(y_pred_valid),np.var(y_pred_valid)))

        try:
            special_loss = loss_fn[self.args.loss](y_pred,valid_pheno).numpy()
            print("The estimated loss defined by model loss function as {} is: ".format(self.args.loss),special_loss)
        except:
            print("The model didn't used the special loss function (e.g. R2) or faced some issues..")
            special_loss = mse
        

        print("Validate prediction accuracy (measured as Pearson's correlation) is: ",
              accuracy_valid)
        return accuracy_valid,mse,special_loss

    def export_record(self):

        self.record.to_csv("{}/{}_train_record_{}.csv".format(os.path.abspath(self.args.output), self.model_name, self.args.trait), sep="\t")
        print("Result:")
        print(self.record)

        return

class Model:

    def __init__(self,args):

        self.args = args
        self._init_model = NN(args)
        self.lr_schedule = self._init_model.lr_schedule
        self._data_requirements = None
        self.modelling = None
        self.data_transform = None

    def get_model_name(self):
        return self._init_model.model_name()

    def init_model(self):

        self._init_model = MODELS[self.args.model](self.args)
        self.data_transform = self._init_model.data_transform
        self.modelling = self._init_model.model
        self.lr_schedule = self._init_model.lr_schedule

        return

    def load_model(self,path):

        self._init_model = MODELS[self.args.model](self.args)
        self.data_transform = self._init_model.data_transform
        self.modelling = self._init_model.model
        self.modelling = keras.models.load_model(path)
        self.lr_schedule = self._init_model.lr_schedule

        #self._init_model = load(path)
        return


def get_model_summary(model: tf.keras.Model) -> str:
    string_list = []
    model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)

def main():
    print("Test mode:")
    args = get_args()
    
    '''
    config_path = os.path.abspath(args.config)
    print("Get config file path from: ", config_path)
    config = configparser.ConfigParser()
    if platform.system().lower() == "windows":
        print(config_path)
        config.read(config_path)
    else:
        config.read(config_path)
    '''

    """
    Create folders from given output path
    """
    if args.output[0] == "/":
        locat = '/' + args.output.strip('/') + '/'
    else:
        locat = args.output.strip('/') + '/'
    if not os.path.exists(locat):
        os.mkdir(locat)

    ##write args to a txt file in the locat
    with open(locat + 'args.txt', 'w') as f:
        f.write(str(args))

    
    if args.build is True:
        composer = ML_composer(args=args)
        composer.get_data(configer=None,args=args)

        index_ref = composer.prepare_cross_validate()
        i = 1
        for train_idx,valid_idx in index_ref:
            print("Cross-validate: {},{}".format(i,valid_idx[0]))
            composer.prepare_training(train_idx,valid_idx)
            composer.compose(train_idx,valid_idx,valid_idx[0])
            i+=1
    elif args.analysis is True and args.load is not None and args.build is False:
        print("Start analysis model...")
        investigate_model(
                        model_path=args.load,ploidy=args.ploidy,args=self.args)
    #composer.prepare_model()


    

if __name__ == "__main__":
    main()




