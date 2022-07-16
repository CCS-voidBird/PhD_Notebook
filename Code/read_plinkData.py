import pandas as pd
import numpy as np
import configparser
import argparse
import platform
import psutil
import os

def read_ped(path,flags):

    data = pd.read_csv(path,sep="\t")
    fid = data.iloc[:,0]
    iid = data.iloc[:,1]
    parents = data.iloc[:,2:4]
    sex = data.iloc[:,4]
