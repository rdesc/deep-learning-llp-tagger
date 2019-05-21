import numpy as np
import seaborn as sns
from column_definition import *

import pandas as pd

import uproot

import concurrent.futures
import multiprocessing

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

def plot_three_histos(signal,qcd,bib,name,xmin,xmax,bins):
    plt.hist(signal, range=(xmin,xmax),  density=True, color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=bins,label="Signal")
    #plt.hist(qcd, range=(xmin,xmax), density=True, color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="QCD")
    plt.hist(bib, range=(xmin,xmax), density=True, color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="BIB")
    plt.xlabel(name)
    plt.legend()
    plt.savefig("plots/" + name+ ".pdf", format='pdf', transparent=True)
    plt.clf()

def plot_constit(x_constit,y_constit,z_constit,name):


    #print(x_constit.shape)
    #print(y_constit.shape)
    print(len(z_constit))
    cmap_sig = sns.cubehelix_palette(rot=-.4,dark=0, light=1,as_cmap=True) #green

    plt.figure()
    plt.hist2d(x_constit, y_constit,
               bins=[16, 16],
               range=[[-1.5, 1.5], [-3.14, 3.14]],
               #range = [[-math.pi,math.pi],[-math.pi,math.pi]],
               #norm=LogNorm(),
               #weights=z_constit,
               cmap = cmap_sig)
    cbar = plt.colorbar()
    #cbar.ax.set_ylabel(r'Jet p$_\mathrm{T}$ per pixel [GeV]')
    cbar.ax.set_ylabel(r'Normalized # clusters')
    plt.xlabel("Pseudorapidity $\eta$")
    plt.ylabel("Azimuthal angle $\phi$")

    plt.savefig("plots/" + name + ".pdf", format='pdf', transparent=True)
    plt.clf()

def do_plotting(signal,qcd,bib,name,xmin,xmax,bins):

    filter_nn_MSeg = [col for col in signal if col.startswith(name)]

    signal_MSeg = remove_values_from_list(signal[filter_nn_MSeg].values.flatten(),0)
    qcd_MSeg = remove_values_from_list(qcd[filter_nn_MSeg].values.flatten(),0)
    bib_MSeg = remove_values_from_list(bib[filter_nn_MSeg].values.flatten(),0)


    plot_three_histos(signal_MSeg,qcd_MSeg,bib_MSeg,name,xmin,xmax,bins)

def plot_vars(data):

    signal = data[data.label == 1]
    print(signal.shape[0])
    qcd = data[data.label == 0]
    print(qcd.shape[0])
    bib = data[data.label == 2]
    print(bib.shape[0])

    signal_clus_pt = signal.iloc[:,slice(6,6+28*20,28)]
    qcd_clus_pt = qcd.iloc[:,slice(6,6+28*20,28)]
    bib_clus_pt = bib.iloc[:,slice(6,6+28*20,28)]

    signal_clus_eta = signal.iloc[:,slice(7,7+28*20,28)]
    qcd_clus_eta = qcd.iloc[:,slice(7,7+28*20,28)]
    bib_clus_eta = bib.iloc[:,slice(7,7+28*20,28)]

    signal_clus_phi = signal.iloc[:,slice(8,8+28*20,28)]
    qcd_clus_phi = qcd.iloc[:,slice(8,8+28*20,28)]
    bib_clus_phi = bib.iloc[:,slice(8,8+28*20,28)]

    signal_all_clus_pt = remove_values_from_list(signal_clus_pt.values.flatten(),0)
    qcd_all_clus_pt = remove_values_from_list(qcd_clus_pt.values.flatten(),0)
    bib_all_clus_pt = remove_values_from_list(bib_clus_pt.values.flatten(),0)

    signal_all_clus_eta = remove_values_from_list(signal_clus_eta.values.flatten(),0)
    qcd_all_clus_eta = remove_values_from_list(qcd_clus_eta.values.flatten(),0)
    bib_all_clus_eta = remove_values_from_list(bib_clus_eta.values.flatten(),0)

    signal_all_clus_phi = remove_values_from_list(signal_clus_phi.values.flatten(),0)
    qcd_all_clus_phi = remove_values_from_list(qcd_clus_phi.values.flatten(),0)
    bib_all_clus_phi = remove_values_from_list(bib_clus_phi.values.flatten(),0)
 
    plot_constit(signal_all_clus_eta,signal_all_clus_phi, signal_all_clus_pt, "signal_constits")
    plot_constit(bib_all_clus_eta,bib_all_clus_phi, bib_all_clus_pt, "bib_constits")

    do_plotting(signal,qcd,bib,"clus_pt",0,60000,30)
    do_plotting(signal,qcd,bib,"clus_eta",-2.5,2.5,20)
    do_plotting(signal,qcd,bib,"clus_phi",-3.14,3.14,20)

    do_plotting(signal,qcd,bib,"jet_pt",0,200000,50)
    do_plotting(signal,qcd,bib,"jet_eta",-2.5,2.5,20)
    do_plotting(signal,qcd,bib,"jet_phi",-3.14,3.14,20)

    do_plotting(signal,qcd,bib,"nn_MSeg_etaDir",-8,8,20)
    do_plotting(signal,qcd,bib,"nn_MSeg_etaPos",-8,8,20)


    do_plotting(signal,qcd,bib,"e_TileBar0",0,10000,20)
    do_plotting(signal,qcd,bib,"e_TileBar1",0,10000,20)
    do_plotting(signal,qcd,bib,"e_TileBar2",0,10000,20)

    do_plotting(signal,qcd,bib,"nn_track_pt",0,10000,20)
    do_plotting(signal,qcd,bib,"nn_track_eta",-2.5,2.5,20)
    do_plotting(signal,qcd,bib,"nn_track_phi",-3.14,3.14,20)
    do_plotting(signal,qcd,bib,"nn_track_d0",0,4,40)
    do_plotting(signal,qcd,bib,"nn_track_z0",0,300,20)
    do_plotting(signal,qcd,bib,"nn_track_PixelHits",0,10,10)
    do_plotting(signal,qcd,bib,"nn_track_SCTHits",0,10,10)
    

    filter_nn_clus = [col for col in signal if col.startswith("clus_pt")]
    filter_nn_track = [col for col in signal if col.startswith("nn_track_pt")]
    filter_nn_MSeg = [col for col in signal if col.startswith("nn_MSeg_etaDir")]

    plot_three_histos( (signal[filter_nn_clus].astype(bool).sum(axis=1)).values.flatten(), (qcd[filter_nn_clus].astype(bool).sum(axis=1)).values.flatten(), (bib[filter_nn_clus].astype(bool).sum(axis=1)).values.flatten(), "n_constits", 0, 20, 20)
    plot_three_histos( (signal[filter_nn_track].astype(bool).sum(axis=1)).values.flatten(), (qcd[filter_nn_track].astype(bool).sum(axis=1)).values.flatten(), (bib[filter_nn_track].astype(bool).sum(axis=1)).values.flatten(), "n_tracks", 0, 25, 25)
    plot_three_histos( (signal[filter_nn_MSeg].astype(bool).sum(axis=1)).values.flatten(), (qcd[filter_nn_MSeg].astype(bool).sum(axis=1)).values.flatten(), (bib[filter_nn_MSeg].astype(bool).sum(axis=1)).values.flatten(), "n_MuonSegments", 0, 70, 70)





