import matplotlib
import matplotlib as mpl
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter, ScalarFormatter

import numpy as np
import seaborn as sns

import pandas as pd

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

def analyse_roc_benchmark(roc_files):

    myDict = {}

    for item in roc_files:
        file = open(item,"r")
        counter = 0
        for line in file:
            if counter > 0:
                cs_string = line.split(',')
                #print("Dict: " + str(myDict))
                if cs_string[0] in myDict:
                    myDict[cs_string[0]].append(float(cs_string[1]))
                else:
                    myDict[cs_string[0]] = [float(cs_string[1])]
            counter = counter + 1

    for item in myDict:
        standard_dev = np.std(myDict[item])
        list_mean = np.mean(myDict[item])

        print("Bib Eff: " + str(round(-float(item)+1,4)) + ", AUC mean: " + str(round(list_mean,4)) + ", AUC std: " + str(round(standard_dev,4)))
        
def analyse_roc_frac(roc_files):

    myDict = {}
    errors = [0.0307, 0.0236, 0.0255, 0.0131, 0.0044, 0.0008, 0.0004, 0.0004, 0.0004]

    for item in roc_files:
        file = open(item,"r")
        counter = 0
        current_frac = 0
        for line in file:
            if counter == 0:
                cs_string = line.split(',')
                current_frac = float(cs_string[0])
            if counter > 2:
                cs_string = line.split(',')
                #print("Dict: " + str(myDict))
                if cs_string[0] in myDict:
                    myDict[cs_string[0]][0].append(float(cs_string[1]))
                    myDict[cs_string[0]][1].append(float(current_frac))
                else:
                    myDict[cs_string[0]] = [[float(cs_string[1])],[current_frac]]
            counter = counter + 1

    error_counter = 2
    for item in myDict:
        bib_eff = float(item)
        plt.errorbar(myDict[item][1], [1-x for x in myDict[item][0]], yerr = errors[error_counter], label = f"BIB Eff: {(-bib_eff+1):.3f}", fmt='o')
        plt.xlabel("Fraction of 2M events used in training")
        plt.ylabel("1-AUC")
        error_counter = error_counter + 1

    plt.legend()
    plt.yscale('log', nonposy='clip')
    plt.savefig("plots/lstm_fracTest/AUC_analysis_frac.pdf", format='pdf', transparent=True)

def analyse_roc_numMaxConstits(roc_files):

    myDict = {}
    errors = [0.0307, 0.0236, 0.0255, 0.0131, 0.0044, 0.0008, 0.0004, 0.0004, 0.0004]

    for item in roc_files:
        file = open(item,"r")
        counter = 0
        current_frac = 0
        for line in file:
            if counter == 0:
                cs_string = line.split(',')
                current_frac = float(cs_string[1])
            if counter > 2:
                cs_string = line.split(',')
                #print("Dict: " + str(myDict))
                if cs_string[0] in myDict:
                    myDict[cs_string[0]][0].append(float(cs_string[1]))
                    myDict[cs_string[0]][1].append(float(current_frac))
                else:
                    myDict[cs_string[0]] = [[float(cs_string[1])],[current_frac]]
            counter = counter + 1

    error_counter = 2
    for item in myDict:
        bib_eff = float(item)
        plt.errorbar(myDict[item][1], [1-x for x in myDict[item][0]], yerr = errors[error_counter], label = f"BIB Eff: {(-bib_eff+1):.3f}", fmt='o')
        plt.xlabel("Number of Constituents used")
        plt.ylabel("1-AUC")
        error_counter = error_counter + 1

    plt.legend()
    plt.yscale('log', nonposy='clip')
    plt.savefig("plots/lstm_fracTest/AUC_analysis_numMaxConstits.pdf", format='pdf', transparent=True)


def analyse_roc_constitLSTM(roc_files):

    myDict = {}
    errors = [0.0307, 0.0236, 0.0255, 0.0131, 0.0044, 0.0008, 0.0004, 0.0004, 0.0004]

    for item in roc_files:
        file = open(item,"r")
        counter = 0
        current_frac = 0
        for line in file:
            if counter == 0:
                cs_string = line.split(',')
                current_frac = float(cs_string[1])
            if counter > 2:
                cs_string = line.split(',')
                #print("Dict: " + str(myDict))
                if cs_string[0] in myDict:
                    myDict[cs_string[0]][0].append(float(cs_string[1]))
                    myDict[cs_string[0]][1].append(float(current_frac))
                else:
                    myDict[cs_string[0]] = [[float(cs_string[1])],[current_frac]]
            counter = counter + 1

    error_counter = 2
    for item in myDict:
        bib_eff = float(item)
        plt.errorbar(myDict[item][1], [1-x for x in myDict[item][0]], yerr = errors[error_counter], label = f"BIB Eff: {(-bib_eff+1):.3f}", fmt='o')
        plt.xlabel("Number of Constituents used")
        plt.ylabel("1-AUC")
        error_counter = error_counter + 1

    plt.legend()
    plt.yscale('log', nonposy='clip')
    plt.savefig("plots/lstm_fracTest/AUC_analysis_numMaxConstits.pdf", format='pdf', transparent=True)

def analyse_roc(roc_files,string_int,name):

    myDict = {}
    errors = [0.0307, 0.0236, 0.0255, 0.0131, 0.0044, 0.0008, 0.0004, 0.0004, 0.0004]

    for item in roc_files:
        file = open(item,"r")
        counter = 0
        current_frac = 0
        for line in file:
            if counter == 0:
                cs_string = line.split(',')
                current_frac = float(cs_string[string_int])
            if counter > 2:
                cs_string = line.split(',')
                print(cs_string[0])
                #print("Dict: " + str(myDict))
                if cs_string[0] in myDict:
                    myDict[cs_string[0]][0].append(float(cs_string[1]))
                    myDict[cs_string[0]][1].append(float(current_frac))
                else:
                    myDict[cs_string[0]] = [[float(cs_string[1])],[current_frac]]
            counter = counter + 1

    error_counter = 2
    for item in myDict:
        bib_eff = float(item)
        plt.errorbar(myDict[item][1], [1-x for x in myDict[item][0]], yerr = errors[error_counter], label = f"BIB Eff: {(-bib_eff+1):.3f}", fmt='o')
        plt.xlabel(name)
        plt.ylabel("1-AUC")
        error_counter = error_counter + 1

    plt.legend()
    plt.yscale('log', nonposy='clip')
    plt.savefig("plots/lstm_fracTest/AUC_analysis_"+name+".pdf", format='pdf', transparent=True)

def analyse_roc_inclusion(roc_files,string_int,name):

    myDict = {}
    errors = [0.0307, 0.0236, 0.0255, 0.0131, 0.0044, 0.0008, 0.0004, 0.0004, 0.0004]
    dummy_dict = {}

    for item in roc_files:
        file = open(item,"r")
        counter = 0
        current_counter = 0
        current_frac = ""
        if "dense" in item:
            current_frac = "dense only"
            current_counter = 0
        elif "doTrackLSTM_False_doMSegLSTM_False" in item:
            current_frac = "lstm constits only"
            current_counter = 1
        elif "doTrackLSTM_True_doMSegLSTM_False" in item:
            current_frac = "lstm constits + tracks"
            current_counter = 2
        elif "doTrackLSTM_False_doMSegLSTM_True" in item:
            current_frac = "lstm constits + MSegs"
            current_counter = 3
        elif "doTrackLSTM_True_doMSegLSTM_True" in item:
            current_frac = "lstm all systems"
            current_counter = 4
        print(current_frac)
        line_counter=0
        for line in file:
            if counter > 2:
                cs_string = line.split(',')
                print(cs_string[0])
                #print("Dict: " + str(myDict))
                if cs_string[0] in myDict:
                    myDict[cs_string[0]][0].append(float(cs_string[1]))
                    myDict[cs_string[0]][1].append((current_frac))
                    dummy_dict[cs_string[0]][0].append(current_counter)
                    dummy_dict[cs_string[0]][1].append((current_frac))
                else:
                    myDict[cs_string[0]] = [[float(cs_string[1])],[current_frac]]
                    dummy_dict[cs_string[0]] = [[current_counter],[current_frac]]
                line_counter = line_counter+1
            counter = counter + 1

    error_counter = 2
    for item in myDict:
        bib_eff = float(item)
        print(dummy_dict[item][1])
        print(dummy_dict[item][0])
        plt.xticks(dummy_dict[item][0], dummy_dict[item][1])
        plt.errorbar(dummy_dict[item][0], [1-x for x in myDict[item][0]], yerr = errors[error_counter], label = f"BIB Eff: {(bib_eff):.3f}", fmt='o')
        plt.xlabel(name)
        plt.ylabel("1-AUC")
        error_counter = error_counter + 1

    plt.legend()
    plt.yscale('log', nonposy='clip')
    plt.savefig("plots/lstm_fracTest/AUC_analysis_"+name+".pdf", format='pdf', transparent=True)


def analyse_roc_signalBenchmark(roc_files,string_int):

    myDict = {}

    for item in roc_files:
        file = open(item,"r")
        counter = 0
        for line in file:
            if counter > 0:
                cs_string = line.split(',')
                if ( len(cs_string) == 3 ):
                    #print("Dict: " + str(myDict))
                    temp_key = (cs_string[0], cs_string[1])
                    if temp_key in myDict:
                        myDict[temp_key].append(float(cs_string[2]))
                    else:
                        myDict[temp_key] = [float(cs_string[2])]
            counter = counter + 1

    for item in myDict:
        standard_dev = np.std(myDict[item])
        list_mean = np.mean(myDict[item])

        print("mH: " + str(item[0]) + ", mS: " + str(item[1]) + ", Avg Efficiency: " + str(list_mean) + ", Std Dev: " + str(standard_dev))

