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
    plt.savefig("plots/" + name+ ".png", format='png', transparent=False)
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

    plt.savefig("plots/" + name + ".png", format='png', transparent=False)
    plt.clf()

def do_plotting(signal,qcd,bib,name,xmin,xmax,bins):

    filter_nn_MSeg = [col for col in signal if col.startswith(name)]

    signal_MSeg = remove_values_from_list(signal[filter_nn_MSeg].values.flatten(),np.nan)
    qcd_MSeg = remove_values_from_list(qcd[filter_nn_MSeg].values.flatten(),np.nan)
    bib_MSeg = remove_values_from_list(bib[filter_nn_MSeg].values.flatten(),np.nan)


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

    signal_all_clus_pt = remove_values_from_list(signal_clus_pt.values.flatten(),np.nan)
    qcd_all_clus_pt = remove_values_from_list(qcd_clus_pt.values.flatten(),np.nan)
    bib_all_clus_pt = remove_values_from_list(bib_clus_pt.values.flatten(),np.nan)

    signal_all_clus_eta = remove_values_from_list(signal_clus_eta.values.flatten(),np.nan)
    qcd_all_clus_eta = remove_values_from_list(qcd_clus_eta.values.flatten(),np.nan)
    bib_all_clus_eta = remove_values_from_list(bib_clus_eta.values.flatten(),np.nan)

    signal_all_clus_phi = remove_values_from_list(signal_clus_phi.values.flatten(),np.nan)
    qcd_all_clus_phi = remove_values_from_list(qcd_clus_phi.values.flatten(),np.nan)
    bib_all_clus_phi = remove_values_from_list(bib_clus_phi.values.flatten(),np.nan)
 
    plot_constit(signal_all_clus_eta,signal_all_clus_phi, signal_all_clus_pt, "signal_constits")
    plot_constit(bib_all_clus_eta,bib_all_clus_phi, bib_all_clus_pt, "bib_constits")

    xmin_dict = {"jet_pt":0, "jet_eta":-2.5, "jet_phi":-3.14, "jet_E":0, "clus_pt":0,"clus_eta":-2.5,"clus_phi":-3.14,"e_PreSamplerB":0,"e_EMB1":0,"e_EMB2":0,"e_EMB3":0,"e_PreSamplerE":0,"e_EME1":0,"e_EME2":0,"e_EME3":0,"e_HEC0":0,"e_HEC1":0,"e_HEC2":0,"e_HEC3":0,"e_TileBar0":0,"e_TileBar1":0,"e_TileBar2":0,"e_TileGap1":0,"e_TileGap2":0,"e_TileGap3":0,"e_TileExt0":0,"e_TileExt1":0,"e_TileExt2":0,"e_FCAL0":0,"e_FCAL1":0,"e_FCAL2":0,"clusTime":-10,"nn_track_pt":0,"nn_track_eta":-2.5,"nn_track_phi":-3.14,"nn_track_d0":0,"nn_track_z0":0,"nn_track_PixelShared":-1,"nn_track_PixelSplit":-1,"nn_track_SCTShared":-1,"nn_track_PixelHoles":-1,"nn_track_SCTHoles":-1,"nn_track_PixelHits":-1,"nn_track_SCTHits":-1,"nn_MSeg_etaPos":-4,"nn_MSeg_phiPos":-3.14,"nn_MSeg_etaDir":-8,"nn_MSeg_phiDir":-3.14,"nn_MSeg_t0":-10}

    xmax_dict = {"jet_pt":200000, "jet_eta":2.5, "jet_phi":3.14, "jet_E":200000, "clus_pt":60000,"clus_eta":2.5,"clus_phi":3.14,"e_PreSamplerB":2000,"e_EMB1":4000,"e_EMB2":6000,"e_EMB3":3000,"e_PreSamplerE":100,"e_EME1":300,"e_EME2":300,"e_EME3":300,"e_HEC0":100,"e_HEC1":100,"e_HEC2":100,"e_HEC3":100,"e_TileBar0":2000,"e_TileBar1":2000,"e_TileBar2":2000,"e_TileGap1":100,"e_TileGap2":100,"e_TileGap3":100,"e_TileExt0":100,"e_TileExt1":100,"e_TileExt2":100,"e_FCAL0":100,"e_FCAL1":100,"e_FCAL2":100,"clusTime":10,"nn_track_pt":10000,"nn_track_eta":2.5,"nn_track_phi":3.14,"nn_track_d0":4,"nn_track_z0":300,"nn_track_PixelShared":10,"nn_track_PixelSplit":10,"nn_track_SCTShared":10,"nn_track_PixelHoles":10,"nn_track_SCTHoles":10,"nn_track_PixelHits":10,"nn_track_SCTHits":10,"nn_MSeg_etaPos":4,"nn_MSeg_phiPos":3.14,"nn_MSeg_etaDir":8,"nn_MSeg_phiDir":3.14,"nn_MSeg_t0":10}

    bin_dict = {"jet_pt":40, "jet_eta":20, "jet_phi":20, "jet_E":40, "clus_pt":40,"clus_eta":20,"clus_phi":20,"e_PreSamplerB":20,"e_EMB1":20,"e_EMB2":20,"e_EMB3":20,"e_PreSamplerE":20,"e_EME1":20,"e_EME2":20,"e_EME3":20,"e_HEC0":20,"e_HEC1":20,"e_HEC2":20,"e_HEC3":20,"e_TileBar0":20,"e_TileBar1":20,"e_TileBar2":20,"e_TileGap1":20,"e_TileGap2":20,"e_TileGap3":20,"e_TileExt0":20,"e_TileExt1":20,"e_TileExt2":20,"e_FCAL0":20,"e_FCAL1":20,"e_FCAL2":20,"clusTime":20,"nn_track_pt":40,"nn_track_eta":20,"nn_track_phi":20,"nn_track_d0":30,"nn_track_z0":30,"nn_track_PixelShared":11,"nn_track_PixelSplit":11,"nn_track_SCTShared":11,"nn_track_PixelHoles":11,"nn_track_SCTHoles":11,"nn_track_PixelHits":11,"nn_track_SCTHits":11,"nn_MSeg_etaPos":20,"nn_MSeg_phiPos":20,"nn_MSeg_etaDir":20,"nn_MSeg_phiDir":20,"nn_MSeg_t0":20}

    for key in xmin_dict:
       do_plotting(signal,qcd,bib,key,xmin_dict[key],xmax_dict[key],bin_dict[key])
 
 
    '''
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
    '''

    filter_nn_clus = [col for col in signal if col.startswith("clus_pt")]
    filter_nn_track = [col for col in signal if col.startswith("nn_track_pt")]
    filter_nn_MSeg = [col for col in signal if col.startswith("nn_MSeg_etaDir")]

    plot_three_histos( ( (signal[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_constits", 0, 30, 30)
    plot_three_histos( (  (signal[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_tracks", 0, 20, 20)
    plot_three_histos( ( (signal[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_MuonSegments", 0, 70, 70)





