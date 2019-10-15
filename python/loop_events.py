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

def loop_events(signal_path,qcd_path,bib_path,output_path, isTraining):

    signal_files = [
        glob.glob(signal_path+"/*/*.trees.root"),
    ]
    qcd_files = [
        glob.glob(qcd_path+"/*/*.trees.root"),
    ]
    bib_files = [
        glob.glob(bib_path+"/*/*.trees.root")
    ]

    creation_time = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S_%f/'))
    print("CREATION TIME: " + creation_time)

    s_counter=0
    q_counter=0
    b_counter=0
    numProc="qstat |  wc -l"
    output_path = output_path + "/" + creation_time
    for files_at_mass_point in signal_files:
        for filename in files_at_mass_point:
            talk=""
            if isTraining:
                talk = ('qsub -l nodes=1:ppn=4 -vinput1=\"' + filename + '\",input2=\"' + output_path + '\",input3=\"' + '1' + '\" batch_extract_info.pbs')
            else:
                talk = ('qsub -l nodes=1:ppn=4 -vinput1=\"' + filename + '\",input2=\"' + output_path + '\",input3=\"' + '1' + '\" batch_extract_info_cr2.pbs')
            files_at_once=80
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

    for files_at_mass_point in qcd_files:
        for filename in files_at_mass_point:
            print("FILE: " + str(filename))
            talk=""
            if isTraining:
                talk = ('qsub -l nodes=1:ppn=4 -vinput1=\"' + filename + '\",input2=\"' + output_path + '\",input3=\"' + '1' + '\" batch_extract_info.pbs')
            else:
                talk = ('qsub -l nodes=1:ppn=4 -vinput1=\"' + filename + '\",input2=\"' + output_path + '\",input3=\"' + '1' + '\" batch_extract_info_cr2.pbs')
            files_at_once=80
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

    for files_at_mass_point in bib_files:
        for filename in files_at_mass_point:
            talk=""
            if isTraining:
                talk = ('qsub -l nodes=1:ppn=4 -vinput1=\"' + filename + '\",input2=\"' + output_path + '\",input3=\"' + '1' + '\" batch_extract_info.pbs')
            else:
                talk = ('qsub -l nodes=1:ppn=4 -vinput1=\"' + filename + '\",input2=\"' + output_path + '\",input3=\"' + '1' + '\" batch_extract_info_cr2.pbs')
            files_at_once=80
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





