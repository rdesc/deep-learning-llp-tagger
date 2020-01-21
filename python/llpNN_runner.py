import numpy as np
import seaborn as sns

import pandas as pd

import sys

import glob

import time
import re

import argparse
import subprocess

from gridSearch_train_keras import *
from make_final_plots import *

import gc

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--file_name')
parser.add_argument('--finalPlots_model')
parser.add_argument('--doTraining',action="store_true")
parser.add_argument('--useGPU2',action="store_true")
parser.add_argument('--makeFinalPlots',action="store_true")
args = parser.parse_args(['--file_name','foo','@args.txt'])
args = parser.parse_args(['--finalPlots_model','foo','@args.txt'])


model_to_do_init = "lstm_parametrization_mHonly_batchsize5k_smallConstitWeightInLoss_test"
num_constits_list = [30, 28, 26, 22, 16, 12, 8]
num_tracks_list = [20, 15, 10, 5]

num_constits_lstm = [60, 120, 240]
num_tracks_lstm = [60, 120, 240]
num_msegs_lstm = [25, 50, 200]
lr_values = [0.00005]
name_list = ["processed_output_Lxy1500_Lz3000"]
frac_list = [0.2,0.4,0.6,0.8]
layers_list = [1,2]
node_list = [150,300]
hidden_layer_fractions = [2,3,4]

#train_llp(args.file_name, model_to_do = model_to_do, frac=0.4)
if (args.doTraining == True):
    filename = name_list[0]
    model_to_do = model_to_do_init + filename
    filename = filename+ ".pkl"
    print ( model_to_do)
    train_llp(args.file_name + filename, args.useGPU2, model_to_do = model_to_do, num_constit_lstm = 150, num_track_lstm = 150, num_mseg_lstm = 150, learning_rate = 0.00005, numConstitLayers = 1, numTrackLayers = 1, numMSegLayers = 1, hiddenFraction = 2, epochs = 50, dropout_value = 0.2, reg_value=0.002)
    gc.collect()

if (args.makeFinalPlots == True):
    input_file = args.file_name + "/validation_dec24.pkl"
    plot_vars_final(input_file, model_to_do= args.finalPlots_model, num_constit_lstm = 150, num_track_lstm = 150, num_mseg_lstm = 150, learning_rate = 0.00005, numConstitLayers = 1, numTrackLayers = 1, numMSegLayers = 1, hiddenFraction = 2, epochs = 200, dropout_value = 0.2, reg_value=0.002),

