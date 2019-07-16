import numpy as np
import seaborn as sns

import pandas as pd

import uproot

from make_plots import *
from flatten_dist import *
from pre_process import *
from parametrize_masses import *
from loop_events import *
from combine_files import *
from keras_loop import *

import concurrent.futures
import multiprocessing

import itertools

import sys

import glob

import time
import re

import argparse
import subprocess

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--signal_path')
parser.add_argument('--qcd_path')
parser.add_argument('--bib_path')
parser.add_argument('--output_path')
parser.add_argument('--output_combo_path')
parser.add_argument('--input_combo_path')
parser.add_argument('--input_training_path')
parser.add_argument('--output_process_path')
parser.add_argument('--doTransform',action="store_true")
parser.add_argument('--doCombination',action="store_true")
parser.add_argument('--doProcessing',action="store_true")
parser.add_argument('--makePlots',action="store_true")
parser.add_argument('--doTraining',action="store_true")
args = parser.parse_args(['--signal_path','foo','@args.txt',
                   '--qcd_path','foo','@args.txt',
                   '--signal_path','foo','@args.txt',
                   '--output_path','foo','@args.txt',
                   '--input_combo_path','foo','@args.txt',
                   '--output_process_path','foo','@args.txt',
                   '--input_training_path','foo','@args.txt',
                   '--output_combo_path','foo','@args.txt'])


min_pt = 40000
max_pt = 300000
files_at_once = 80
start = time.time()

df = pd.DataFrame()

if args.doTransform:
    print("Doing transform")
    loop_events(args.signal_path,args.qcd_path,args.bib_path,args.output_path)

if args.doCombination:
    print("Doing Combination in folder: " + str(args.input_combo_path))
    df = combine_files(args.input_combo_path)

if args.doProcessing:

    if not (args.doCombination):
        df = pd.read_pickle(args.input_combo_path + "/combine_output.pkl")
    if (args.makePlots):
        print("Plotting...")
        print("It has been " + (str(time.time() - start)) + "seconds since start")
        plot_vars(df)
        #plot_vars(df, prefix="_cleanJets")
    print("Flattening...")
    min_pt = 40000
    max_pt = 300000
    print("It has been " + (str(time.time() - start)) + "seconds since start")
    df = flatten(df, min_pt, max_pt, 20)
    print("pre-processing...")
    print("It has been " + (str(time.time() - start)) + "seconds since start")
    df = pre_process(df, 0, max_pt)
    if (args.makePlots):
        plot_vars(df, prefix="_post_processing")
    df = parametrize_masses(df)
    print("Saving processed file...")
    print("It has been " + (str(time.time() - start)) + "seconds since start")
    df.to_pickle(args.output_process_path + "processed_output.pkl")

if args.doTraining:
    eventFractions = [1.0, 0.8, 0.6, 0.4, 0.2]
    list_max_constits = [30, 25, 20, 12, 8]
    list_max_tracks  = [20, 15, 10, 8, 5]
    list_max_MSegs = [70, 50, 30, 20, 10]
    #list_constit_lstm = [10, 10, 10, 100, 100, 100, 1000, 1000, 1000]
    #list_track_lstm = [10, 100, 10000, 10, 100, 1000, 10, 100, 1000]
    #list_mseg_lstm = [10, 10, 100, 100, 1000, 1000, 10, 100, 1000]
    inputFile = args.input_training_path + 'processed_output.pkl'
    keras_loop(inputFile , fracTraining = eventFractions, num_max_constits = list_max_constits, num_max_tracks = list_max_tracks , num_max_MSegs = list_max_MSegs )
    












