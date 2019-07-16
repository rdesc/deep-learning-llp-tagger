
import numpy as np
import seaborn as sns

import pandas as pd

import itertools

import sys

import glob

import time
import re

import argparse
import subprocess
import random

from random import shuffle

def combine_files(output_path):
    signal_files = [
        glob.glob(output_path+"/signal*.pkl"),
    ]
    qcd_files = [
        glob.glob(output_path+"/qcd*.pkl"),
    ]
    bib_files = [
        glob.glob(output_path+"/bib*.pkl"),
    ]
    random.seed(24)
    shuffle(qcd_files[0])
    shuffle(bib_files[0])

    df = pd.DataFrame()

    signal_length=0
    qcd_length=0
    bib_length=0

    for files_at_mass_point in signal_files:
        for filename in files_at_mass_point:
            temp = pd.read_pickle(filename)
            signal_length = signal_length + temp.shape[0]
            print("Signal length: " + str(signal_length))
            df = df.append(temp, ignore_index=False)

    print("Signal length: " + str(signal_length))

    for files_at_mass_point in qcd_files:
        for filename in files_at_mass_point:
            print(filename)
            temp = pd.read_pickle(filename)
            qcd_length = qcd_length + temp.shape[0]
            df = df.append(temp, ignore_index=False)
            if qcd_length > signal_length:
                break
        if qcd_length > signal_length:
            break

    print("QCD length: " + str(qcd_length))

    for files_at_mass_point in bib_files:
        for filename in files_at_mass_point:
            temp = pd.read_pickle(filename)
            bib_length = bib_length + temp.shape[0]
            df = df.append(temp, ignore_index=False)
            if bib_length > signal_length:
                break
        if bib_length > signal_length:
            break

    print("BIB length: " + str(bib_length))


    df.to_pickle(output_path + "/combine_output.pkl") 

    return df

