import numpy as np
import seaborn as sns

import pandas as pd

import uproot

from make_plots import *
from flatten_dist import *
from pre_process import *
from parametrize_masses import *

import concurrent.futures
import multiprocessing

import itertools

import sys

import glob

import time
import re

import argparse
import subprocess

from datetime import datetime

def keras_loop(inputFile , fracTraining = [1.0], num_max_constits = [30], num_max_tracks = [25], num_max_MSegs = [70]):


    creation_time = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S/'))
    numProc="qstat |  wc -l"
    files_at_once=21

    for frac in fracTraining:
        nameString = "lstm_fracEvents_" + str(frac) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(num_max_MSegs[0]) + "_" + creation_time
        talk = ('qsub  -l walltime=12:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(frac) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + nameString  + '\" batch_train_gridSearch.pbs')
        print(talk)
        print(files_at_once)
        subprocess.call(talk, shell=True)
        numJobs = subprocess.check_output(numProc, shell=True)
        print("num jobs: " + str(int(numJobs)) )
        time.sleep(0.1)
        while (int(numJobs) > files_at_once):
           #print("waiting...")
           time.sleep(10)

    for max_constits in num_max_constits:
        nameString = "lstm_fracEvents_" + str(fracTraining[0]) + "_constits_" + str(max_constits) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(num_max_MSegs[0]) + "_" + creation_time
        talk = ('qsub  -l walltime=12:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(max_constits) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + nameString  + '\" batch_train_gridSearch.pbs')
        print(talk)
        print(files_at_once)
        subprocess.call(talk, shell=True)
        numJobs = subprocess.check_output(numProc, shell=True)
        print("num jobs: " + str(int(numJobs)) )
        time.sleep(0.1)
        while (int(numJobs) > files_at_once):
           #print("waiting...")
           time.sleep(10)
           numJobs = subprocess.check_output(numProc, shell=True)
           numJobs = subprocess.check_output(numProc, shell=True)


    for max_tracks in num_max_tracks:
        nameString = "lstm_fracEvents_" + str(fracTraining[0]) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(max_tracks) + "_MSegs_" + str(num_max_MSegs[0]) + "_" + creation_time
        talk = ('qsub  -l walltime=12:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(max_tracks) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + nameString  + '\" batch_train_gridSearch.pbs')
        print(talk)
        print(files_at_once)
        subprocess.call(talk, shell=True)
        numJobs = subprocess.check_output(numProc, shell=True)
        print("num jobs: " + str(int(numJobs)) )
        time.sleep(0.1)
        while (int(numJobs) > files_at_once):
           #print("waiting...")
           time.sleep(10)
           numJobs = subprocess.check_output(numProc, shell=True)

    for max_MSegs in num_max_MSegs:
        nameString = "lstm_fracEvents_" + str(fracTraining[0]) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(max_MSegs) + "_" + creation_time
        talk = ('qsub  -l walltime=12:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(max_MSegs) + '\",input6=\"' + nameString  + '\" batch_train_gridSearch.pbs')
        print(talk)
        print(files_at_once)
        subprocess.call(talk, shell=True)
        numJobs = subprocess.check_output(numProc, shell=True)
        print("num jobs: " + str(int(numJobs)) )
        time.sleep(0.1)
        while (int(numJobs) > files_at_once):
           #print("waiting...")
           time.sleep(10)
           numJobs = subprocess.check_output(numProc, shell=True)





