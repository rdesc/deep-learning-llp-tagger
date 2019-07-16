
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

from random import shuffle
import matplotlib
import matplotlib as mpl
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors


def plot_one_histo(qcd,name,xmin,xmax,bins, prefix):
    fig,ax = plt.subplots()
    ax.hist(qcd, range=(xmin,xmax), density=True, color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="QCD")
    ax.set_xlabel(name)
    ax.set_ylabel("Arb. Units")
    ax.legend()

    textstr = prefix

    #matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.8, textstr, color='black', transform=ax.transAxes,
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))


    plt.savefig("plots/checks/" + name+ prefix +".png", format='png', transparent=False)
    plt.clf()
    plt.close()


def do_plotting(qcd,name,xmin,xmax,bins, prefix):

    filter_nn_MSeg = [col for col in qcd if col.startswith(name)]

    #signal_MSeg = remove_values_from_list(signal[filter_nn_MSeg].values.flatten(),np.nan)
    #qcd_MSeg = remove_values_from_list(qcd[filter_nn_MSeg].values.flatten(),np.nan)
    #bib_MSeg = remove_values_from_list(bib[filter_nn_MSeg].values.flatten(),np.nan)

    qcd_MSeg = qcd[filter_nn_MSeg].dropna()


    plot_one_histo(qcd_MSeg.values.flatten(),name,xmin,xmax,bins, prefix)


jz2_files = [
    glob.glob("/data/fcormier/calRatio/fullRun2/transformOutput/2019-07-10_17:18:21_603916/qcd_2*.pkl"),
]
jz3_files = [
    glob.glob("/data/fcormier/calRatio/fullRun2/transformOutput/2019-07-10_17:18:21_603916/qcd_3*.pkl"),
]


df_jz2 = pd.DataFrame()
df_jz3 = pd.DataFrame()
jz2_length=0
jz3_length=0


for files_at_mass_point in jz2_files:
    for filename in files_at_mass_point:
        temp = pd.read_pickle(filename)
        jz2_length = jz2_length + temp.shape[0]
        print("JZ2: " + str(jz2_length))
        df_jz2 = df_jz2.append(temp, ignore_index=False)
        if (jz2_length > 1000000):
            break

do_plotting(df_jz2,"jet_pt",0,300000,40,"JZ2")

for files_at_mass_point in jz3_files:
    for filename in files_at_mass_point:
        temp = pd.read_pickle(filename)
        jz3_length = jz3_length + temp.shape[0]
        print("JZ3: " + str(jz3_length))
        df_jz3 = df_jz3.append(temp, ignore_index=False)
        if (jz3_length > 1000000):
            break

do_plotting(df_jz3,"jet_pt",0,300000,40,"JZ3")

