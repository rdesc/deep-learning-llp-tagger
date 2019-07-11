import numpy as np
import seaborn as sns

import pandas as pd


import itertools

import sys

import matplotlib
import matplotlib as mpl
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def plot_three_histos(signal,s_weights,qcd,qcd_weights,bib,bib_weights,name,xmin,xmax,bins):
    plt.hist(signal, weights = s_weights, range=(xmin,xmax),   color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=bins,label="Signal")
    plt.hist(qcd, weights = qcd_weights, range=(xmin,xmax),  color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="QCD")
    plt.hist(bib, weights = bib_weights, range=(xmin,xmax),  color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="BIB")
    plt.xlabel(name)
    plt.legend()
    plt.savefig("plots/" + name+ "_flat.png", format='png', transparent=False)
    plt.clf()
    plt.close()


def do_plotting(signal,qcd,bib,name,xmin,xmax,bins):
    filter_nn_MSeg = [col for col in signal if col.startswith(name)]

    signal_MSeg = remove_values_from_list(signal[filter_nn_MSeg].values.flatten(),np.nan)
    signal_weights = remove_values_from_list(signal["flatWeight"].values.flatten(),np.nan)
    qcd_MSeg = remove_values_from_list(qcd[filter_nn_MSeg].values.flatten(),np.nan)
    qcd_weights = remove_values_from_list(qcd["flatWeight"].values.flatten(),np.nan)
    bib_MSeg = remove_values_from_list(bib[filter_nn_MSeg].values.flatten(),np.nan)
    bib_weights = remove_values_from_list(bib["flatWeight"].values.flatten(),np.nan)


    plot_three_histos(signal_MSeg,signal_weights,qcd_MSeg,qcd_weights,bib_MSeg,bib_weights,name,xmin,xmax,bins)



def flatten(data, low_bin, high_bin, n_bins):
    
    signal = data[data.label == 1]
    print(signal.shape[0])
    qcd = data[data.label == 0]
    print(qcd.shape[0])
    bib = data[data.label == 2]
    print(bib.shape[0])

    signal = signal.loc[ (signal.jet_pt > low_bin) & (signal.jet_pt < high_bin)].copy()
    qcd = qcd.loc[ (qcd.jet_pt > low_bin) & (qcd.jet_pt < high_bin)].copy()
    bib = bib.loc[ (bib.jet_pt > low_bin) & (bib.jet_pt < high_bin)].copy()

    print(signal.shape[0])
    print(qcd.shape[0])
    print(bib.shape[0])

    print(qcd["mcEventWeight"].sum())

    signal_array, signal_bin_edges = np.histogram(signal["jet_pt"], bins = n_bins, range = [low_bin,high_bin], density = True)
    print(signal_array)
    print(signal_bin_edges)
    print(np.sum(signal_array)*((high_bin-low_bin)/n_bins))

  
    qcd_array, qcd_bin_edges = np.histogram(qcd["jet_pt"], weights=qcd["mcEventWeight"], bins = n_bins, range = [low_bin,high_bin], density = True)
    print(qcd_array)
    print(qcd_bin_edges)
    print(np.sum(qcd_array)*((high_bin-low_bin)/n_bins))

    bib_array, bib_bin_edges = np.histogram(bib["jet_pt"], bins = n_bins, range = [low_bin,high_bin], density = True)
    print(bib_array)
    print(bib_bin_edges)
    print(np.sum(bib_array)*((high_bin-low_bin)/n_bins))

    signal_flatWeights = signal_array*((high_bin-low_bin))
    qcd_flatWeights = qcd_array*((high_bin-low_bin))
    bib_flatWeights = bib_array*((high_bin-low_bin))

    #qcd_correction = signal.loc[ (signal.jet_pt > low_bin) & (signal.jet_pt < high_bin)].shape[0] / ( qcd.loc[ (qcd.jet_pt > low_bin) & (qcd.jet_pt < high_bin)].shape[0])
    qcd_correction = signal.loc[ (signal.jet_pt > low_bin) & (signal.jet_pt < high_bin)].shape[0] / ( qcd["mcEventWeight"].loc[ (qcd.jet_pt > low_bin) & (qcd.jet_pt < high_bin)].sum())
    bib_correction = signal.loc[ (signal.jet_pt > low_bin) & (signal.jet_pt < high_bin)].shape[0] / bib.loc[ (bib.jet_pt > low_bin) & (bib.jet_pt < high_bin)].shape[0]


    for i in range(len(signal_bin_edges)-1):
        signal["flatWeight"].loc[ (signal.jet_pt > signal_bin_edges[i]) & (signal.jet_pt < signal_bin_edges[i+1]) ] = np.ones(signal.loc[ (signal.jet_pt > signal_bin_edges[i]) & (signal.jet_pt < signal_bin_edges[i+1]) ].shape[0]) * 1/(signal_flatWeights[i])
        #qcd["flatWeight"].loc[ (qcd.jet_pt > qcd_bin_edges[i]) & (qcd.jet_pt < qcd_bin_edges[i+1]) ] = np.ones(qcd.loc[ (qcd.jet_pt > qcd_bin_edges[i]) & (qcd.jet_pt < qcd_bin_edges[i+1]) ].shape[0]) * qcd_correction/(qcd_flatWeights[i])
        qcd["flatWeight"].loc[ (qcd.jet_pt > qcd_bin_edges[i]) & (qcd.jet_pt < qcd_bin_edges[i+1]) ] = qcd["mcEventWeight"].loc[ (qcd.jet_pt > qcd_bin_edges[i]) & (qcd.jet_pt < qcd_bin_edges[i+1]) ] * qcd_correction/(qcd_flatWeights[i])
        bib["flatWeight"].loc[ (bib.jet_pt > bib_bin_edges[i]) & (bib.jet_pt < bib_bin_edges[i+1]) ] = np.ones(bib.loc[ (bib.jet_pt > bib_bin_edges[i]) & (bib.jet_pt < bib_bin_edges[i+1]) ].shape[0]) * bib_correction/(bib_flatWeights[i])

    print(np.sum(signal["flatWeight"].loc[ (signal.jet_pt > low_bin) & (signal.jet_pt < high_bin)].values))
    print(np.sum(qcd["flatWeight"].loc[ (qcd.jet_pt > low_bin) & (qcd.jet_pt < high_bin)].values))
    print(np.sum(bib["flatWeight"].loc[ (bib.jet_pt > low_bin) & (bib.jet_pt < high_bin)].values))
    print(signal.loc[ (signal.jet_pt > low_bin) & (signal.jet_pt < high_bin)].shape[0] )

    df = pd.DataFrame()
    df = df.append(signal)
    df = df.append(qcd)
    df = df.append(bib)

    do_plotting(signal,qcd,bib,"jet_pt",low_bin,high_bin,n_bins)

    return(df)

