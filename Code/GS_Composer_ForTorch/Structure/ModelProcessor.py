import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

import numpy as np
from genotypeProcessor.Functions import *
from scipy import stats
from utils.metrics import *

import torch
import torch.nn as nn
from torch import optim
from models import Transformer, LCLFormer, Autoformer
from torchinfo import summary
from torchviz import make_dot

import os
import time

import warnings
"""
This file contains the class for the model that connects the main program and the model
"""

MODELS = {

    "Transformer":Transformer,
    "LCLFormer":LCLFormer,
    "Autoformer":Autoformer
    
}

loss_fn = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    "huber": nn.HuberLoss()
}

    
############################################################################
# Torch based transformer model
# 1. A padding layer to pad the input tensor to the length of kernel size

class Model:

    def __init__(self,args):

        self.args = args
        self.device = self._acqure_device()
        self._init_model = None
        self._data_requirements = None
        self.trait_label = None
        self.early_stopping = None
        self.lr_monitor = None

    def get_model_name(self):
        return self._init_model.model_name()

    def _acqure_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            self.device = torch.device("cpu")

        return device

    def data_transform(self,geno,phenos,anno=None,pheno_standard = False):

        print("USE {} MODEL as training method".format(self.name))
        geno = decoding(geno)
        geno = np.expand_dims(geno, axis=2)
        print("The transformed SNP shape:", geno.shape)
        # for multiple phenos
        if isinstance(phenos, list):
            for i in range(len(phenos)):
                if pheno_standard is True:
                    phenos[i] = stats.zscore(phenos[i])
        return geno,phenos

    def init_model(self):

        self._init_model = MODELS[self.args.model].Model(self.args).float()
        #self.lr_schedule = self._init_model.lr_schedule
        self.trait_label = "MT" if self.args.NumTrait > 1 else self.args.trait[0]
        #self.early_stopping = self._init_model.early_stopping
        #self.lr_monitor = self._init_model.lr_monitor
        if self.args.NumTrait > 1 and self.args.trait is None:
            self.args.trait = ["Trait_{}".format(x) for x in range(self.args.NumTrait)]

        if self.args.use_multi_gpu and self.args.use_gpu:
            self._init_model = nn.DataParallel(self._init_model, device_ids=self.args.device_ids)

        self._init_model.to(self.device)
        return
    

    def _select_criterion(self):
        loss_fn = loss_fn[self.args.loss]
        return loss_fn

    def _select_optimizer(self):
        optimizer = optim.Adam(self._init_model.parameters(), lr=self.args.lr)
        return optimizer

    def _predict(self, geno, phenos, **kwargs):

        #Encoder input: geno
        #Decoder input: phenos

        decoder_input_batch_y = phenos
        encoder_input_batch_x = geno

        #print("Encoder input shape: ", encoder_input.shape)
        #print("Decoder input shape: ", decoder_input.shape)

        def _run_model():
            print("Running model")
            outputs = self._init_model(encoder_input_batch_x, None, decoder_input_batch_y,None)
            if self.args.output_attention:
                outputs, attention = outputs
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        return outputs

    def train(self, train_loader,validate_loader,round=1):

        n_features = train_loader.get_shape()

        path = os.path.join(self.args.output, "round_{}".format(round))
        if not os.path.exists(path):
            os.makedirs(path)

        #print("Got input shape:",n_features)

        
        if round == 1:
            with open(os.path.abspath(self.args.output) + "/model_summary.txt", "w") as fh:
                summary(self._init_model)
            try:
                dot = make_dot(self._init_model.mean(), params=dict(self._init_model.named_parameters()))
                dot.format = 'png'
                dot.render(os.path.abspath(self.args.output) + "/model_graph")
            except:
                "Model plotting function error"

        if self.args.quiet != 0:
            print(summary(self._init_model, input_size=n_features))

        #train_steps = len(train_loader)
        #early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_time_start = time.time()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            self._init_model.train()
            running_loss = 0.0
            iter_count = 0
            train_loss = []
            train_steps = len(train_loader)
            epoch_time = time.time()

            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                model_optim.zero_grad()

                outputs = self._init_model(inputs)
                loss = criterion(outputs, labels)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._init_model(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self._init_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    model_optim.step()

                running_loss += loss.item()    

            metrics = {
                'Train_loss': running_loss,
                'Validation_loss': 0.0,
                'Accuracy': 0.0
            }

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.validation(validate_loader, criterion)
            test_loss = self.validation(train_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            #early_stopping(vali_loss, self.model, path)
            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    break

            #adjust_learning_rate(model_optim, epoch + 1, self.args)

        train_time_end = time.time()
        runtime = train_time_end - train_time_start

        if self.args.save:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        
        return train_loss, runtime


    def validation(self, validate_data_loader, criterion):
        self._init_model.eval()
        prediction = []
        observation = []

        with torch.no_grad():
            for i, data in enumerate(validate_data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self._init_model(inputs)

                prediction.append(outputs.detach().cpu().numpy())
                observation.append(labels.detach().cpu().numpy())

            prediction = np.concatenate(prediction, axis=0)
            observation = np.concatenate(observation, axis=0)

            mse = MSE(prediction, observation)
            r, p = stats.pearsonr(prediction, observation)
            r2 = R2(prediction, observation)


        self._init_model.train()

        return mse, r, p, r2
        
        
        

    def test(self, geno):
        pass

    def load_model(self,path):

        self._init_model = MODELS[self.args.model](self.args)
        self.data_transform = self._init_model.data_transform
        checkpoint = torch.load(path)
        self._init_model.load_state_dict(checkpoint['model_state_dict'])
        #self._init_model = load(path)
        return




def main():
    print("Main function from ClassModel.py")
    #tf.keras.utils.plot_model(model, to_file="./print_model.png", show_shapes=True)

if __name__ == "__main__":
    main()
