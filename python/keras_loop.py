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

def keras_loop(inputFile , fracTraining = [1.0], num_max_constits = [30], num_max_tracks = [20], num_max_MSegs = [70], num_constit_lstm = [12], num_track_lstm = [12], num_mseg_lstm = [5]):


    creation_time = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S/'))
    numProc="qstat |  wc -l"
    files_at_once=20

    #Do  default
    nameString = "lstm__fracEvents_" + str(fracTraining[0]) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(num_max_MSegs[0]) + "_LSTMconstits_" + str(num_constit_lstm[0]) + "_LSTMtracks_" + str(num_track_lstm[0]) + "_LSTMmseg_" + str(num_mseg_lstm[0]) + "_" + creation_time
    talk = ('qsub  -l walltime=48:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + str(num_constit_lstm[0]) + '\",input7=\"'+ str(num_track_lstm[0]) + '\",input8=\"' + str(num_mseg_lstm[0]) + '\",input9=\"' + nameString  + '\" batch_train_gridSearch.pbs')
    print(talk)
    print(files_at_once)
    subprocess.call(talk, shell=True)
    numJobs = subprocess.check_output(numProc, shell=True)
    print("num jobs: " + str(int(numJobs)) )

    if len(fracTraining) > 1:
        for frac in fracTraining[1:]:
            nameString = "lstm__fracEvents_" + str(frac) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(num_max_MSegs[0]) + "_LSTMconstits_" + str(num_constit_lstm[0]) + "_LSTMtracks_" + str(num_track_lstm[0]) + "_LSTMmseg_" + str(num_mseg_lstm[0]) + "_" + creation_time
            talk = ('qsub  -l walltime=48:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(frac) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + str(num_constit_lstm[0]) + '\",input7=\"'+ str(num_track_lstm[0]) + '\",input8=\"' + str(num_mseg_lstm[0]) + '\",input9=\"' + nameString  + '\" batch_train_gridSearch.pbs')
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

    if len(num_max_constits) > 1:
        for max_constits in num_max_constits[1:]:
            nameString = "lstm__fracEvents_" + str(fracTraining[0]) + "_constits_" + str(max_constits) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(num_max_MSegs[0]) + "_LSTMconstits_" + str(num_constit_lstm[0]) + "_LSTMtracks_" + str(num_track_lstm[0]) + "_LSTMmseg_" + str(num_mseg_lstm[0]) + "_" + creation_time
            talk = ('qsub  -l walltime=48:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(max_constits) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + str(num_constit_lstm[0]) + '\",input7=\"'+ str(num_track_lstm[0]) + '\",input8=\"' + str(num_mseg_lstm[0]) + '\",input9=\"' + nameString  + '\" batch_train_gridSearch.pbs')
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


    if len(num_max_tracks) > 1:
        for max_tracks in num_max_tracks[1:]:
            nameString = "lstm__fracEvents_" + str(fracTraining[0]) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(max_tracks) + "_MSegs_" + str(num_max_MSegs[0]) + "_LSTMconstits_" + str(num_constit_lstm[0]) + "_LSTMtracks_" + str(num_track_lstm[0]) + "_LSTMmseg_" + str(num_mseg_lstm[0]) + "_" + creation_time
            talk = ('qsub  -l walltime=48:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(max_tracks) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + str(num_constit_lstm[0]) + '\",input7=\"'+ str(num_track_lstm[0]) + '\",input8=\"' + str(num_mseg_lstm[0]) + '\",input9=\"' + nameString  + '\" batch_train_gridSearch.pbs')
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

    if len(num_max_MSegs) > 1:
        for max_MSegs in num_max_MSegs[1:]:
            nameString = "lstm__fracEvents_" + str(fracTraining[0]) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(max_MSegs) + "_LSTMconstits_" + str(num_constit_lstm[0]) + "_LSTMtracks_" + str(num_track_lstm[0]) + "_LSTMmseg_" + str(num_mseg_lstm[0]) +"_" + creation_time
            talk = ('qsub  -l walltime=48:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(max_MSegs) + '\",input6=\"' + str(num_constit_lstm[0]) + '\",input7=\"'+ str(num_track_lstm[0]) + '\",input8=\"' + str(num_mseg_lstm[0]) + '\",input9=\"' + nameString  + '\" batch_train_gridSearch.pbs')
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

    if len(num_constit_lstm) > 1:
        for lstm_constits in num_constit_lstm[1:]:
            nameString = "lstm__fracEvents_" + str(fracTraining[0]) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(num_max_MSegs[0]) + "_LSTMconstits_" + str(lstm_constits) + "_LSTMtracks_" + str(num_track_lstm[0]) + "_LSTMmseg_" + str(num_mseg_lstm[0]) +"_" + creation_time
            talk = ('qsub  -l walltime=48:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + str(lstm_constits) + '\",input7=\"'+ str(num_track_lstm[0]) + '\",input8=\"' + str(num_mseg_lstm[0]) + '\",input9=\"' + nameString  + '\" batch_train_gridSearch.pbs')
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

    if len(num_track_lstm) > 1:
        for lstm_tracks in num_track_lstm[1:]:
            nameString = "lstm__fracEvents_" + str(fracTraining[0]) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(num_max_MSegs[0]) + "_LSTMconstits_" + str(num_constit_lstm[0]) + "_LSTMtracks_" + str(lstm_tracks) + "_LSTMmseg_" + str(num_mseg_lstm[0]) +"_" + creation_time
            talk = ('qsub  -l walltime=48:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + str(num_constit_lstm[0]) + '\",input7=\"'+ str(lstm_tracks) + '\",input8=\"' + str(num_mseg_lstm[0]) + '\",input9=\"' + nameString  + '\" batch_train_gridSearch.pbs')
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

    if len(num_mseg_lstm) > 1:
        for lstm_msegs in num_mseg_lstm[1:]:
            nameString = "lstm__fracEvents_" + str(fracTraining[0]) + "_constits_" + str(num_max_constits[0]) + "_tracks_" + str(num_max_tracks[0]) + "_MSegs_" + str(num_max_MSegs[0]) + "_LSTMconstits_" + str(num_constit_lstm[0]) + "_LSTMtracks_" + str(num_track_lstm[0]) + "_LSTMmseg_" + str(lstm_msegs) +"_" + creation_time
            talk = ('qsub  -l walltime=48:00:00 -l nodes=1:ppn=16 -vinput1=\"' + inputFile + '\",input2=\"' + str(fracTraining[0]) + '\",input3=\"' + str(num_max_constits[0]) + '\",input4=\"' + str(num_max_tracks[0]) + '\",input5=\"' + str(num_max_MSegs[0]) + '\",input6=\"' + str(num_constit_lstm[0]) + '\",input7=\"'+ str(num_track_lstm[0]) + '\",input8=\"' + str(lstm_msegs) + '\",input9=\"' + nameString  + '\" batch_train_gridSearch.pbs')
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





