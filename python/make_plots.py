import numpy as np
import seaborn as sns
from column_definition import *

import pandas as pd

import uproot

import concurrent.futures
import multiprocessing

import itertools
from itertools import cycle

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

def plot_three_histos(signal,qcd,bib,name,xmin,xmax,bins, prefix):
    fig,ax = plt.subplots()
    ax.hist(signal, range=(xmin,xmax),  density=True, color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=bins,label="Signal")
    ax.hist(qcd, range=(xmin,xmax), density=True, color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="QCD")
    ax.hist(bib, range=(xmin,xmax), density=True, color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="BIB")
    ax.set_xlabel(name)
    ax.set_ylabel("Arb. Units")
    ax.legend()

    textstr = prefix 

    #matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.8, textstr, color='black', transform=ax.transAxes, 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))


    plt.savefig("plots/" + name+ prefix +".png", format='png', transparent=False)
    plt.clf()

def plot_three_histos_withCut(signal,qcd,bib,name,xmin,xmax,bins, prefix, cut_variable, cut_direction, cut_value):
    fig,ax = plt.subplots()
    ax.hist(signal, range=(xmin,xmax),  density=True, color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=bins,label="Signal")
    ax.hist(qcd, range=(xmin,xmax), density=True, color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="QCD")
    ax.hist(bib, range=(xmin,xmax), density=True, color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bins,label="BIB")
    ax.set_xlabel(name)
    ax.legend()
    textstr = cut_variable + " " + cut_direction + " " + str(cut_value) 

    #matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.8, textstr, color='black', transform=ax.transAxes, 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))

    plt.savefig("plots/cutDiagrams/" + name+ prefix + "_" + str(cut_variable) + cut_direction + str(cut_value) +".png", format='png', transparent=False)
    plt.clf()

def plot_constit(x_constit,y_constit,z_constit,name, prefix):


    #print(x_constit.shape)
    #print(y_constit.shape)
    cmap_sig = sns.cubehelix_palette(rot=-.4,dark=0, light=1,as_cmap=True) #green

    range_phi = 3.14
    range_eta = 1.4
    if "process" in prefix:
        range_phi = 1
        range_eta = 1

    plt.figure()
    plt.hist2d(x_constit, y_constit,
               bins=[20, 20],
               range=[[-range_eta, range_eta], [-range_phi, range_phi]],
               #range = [[-math.pi,math.pi],[-math.pi,math.pi]],
               norm=LogNorm(),
               #weights=z_constit,
               cmap = cmap_sig)
    cbar = plt.colorbar()
    #cbar.ax.set_ylabel(r'Jet p$_\mathrm{T}$ per pixel [GeV]')
    cbar.ax.set_ylabel(r'Normalized # clusters')
    plt.xlabel("Pseudorapidity $\eta$")
    plt.ylabel("Azimuthal angle $\phi$")

    plt.savefig("plots/" + name + prefix +".png", format='png', transparent=False)
    plt.clf()

def plot_2d_histos(signal_x, qcd_x, bib_x, name_x, xmin, xmax, x_bins,  signal_y, qcd_y, bib_y, name_y, ymin, ymax, y_bins, prefix): 
    cmap_sig = sns.cubehelix_palette(rot=-.4,dark=0, light=1,as_cmap=True)

    plt.figure()
    plt.hist2d(signal_x, signal_y, bins=[x_bins, y_bins], range=[[xmin,xmax],[ymin,ymax]], norm = LogNorm(), cmap = cmap_sig)  

    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'# Signal')
    plt.xlabel(name_x)
    plt.ylabel(name_y)

    plt.savefig("plots/2Dplots/signal_" + name_x + "_2D_" +  name_y + prefix +".png", format='png', transparent=False)
    plt.clf()

    plt.figure()
    plt.hist2d(qcd_x, qcd_y, bins=[x_bins, y_bins], range=[[xmin,xmax],[ymin,ymax]], norm = LogNorm(), cmap = cmap_sig)  
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'# QCD')
    plt.xlabel(name_x)
    plt.ylabel(name_y)

    plt.savefig("plots/2Dplots/qcd_" + name_x + "_2D_" +  name_y + prefix +".png", format='png', transparent=False)
    plt.clf()

    plt.figure()
    plt.hist2d(bib_x, bib_y, bins=[x_bins, y_bins], range=[[xmin,xmax],[ymin,ymax]], norm = LogNorm(), cmap = cmap_sig)  
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'# BIB')
    plt.xlabel(name_x)
    plt.ylabel(name_y)

    plt.savefig("plots/2Dplots/bib_" + name_x + "_2D_" +  name_y + prefix +".png", format='png', transparent=False)
    plt.clf()

def do_plotting_2D(signal,qcd,bib,name_x,xmin,xmax,x_bins, name_y, ymin,ymax,y_bins, prefix):

    filter_nn_x = [col for col in signal if col.startswith(name_x)]
    filter_nn_y = [col for col in signal if col.startswith(name_y)]

    signal_x = remove_values_from_list(signal[filter_nn_x].values.flatten(),np.nan)
    qcd_x = remove_values_from_list(qcd[filter_nn_x].values.flatten(),np.nan)
    bib_x = remove_values_from_list(bib[filter_nn_x].values.flatten(),np.nan)

    signal_y = remove_values_from_list(signal[filter_nn_y].values.flatten(),np.nan)
    qcd_y = remove_values_from_list(qcd[filter_nn_y].values.flatten(),np.nan)
    bib_y = remove_values_from_list(bib[filter_nn_y].values.flatten(),np.nan)

    if ( (len(signal_x)==len(signal_y)) and (len(qcd_x)==len(qcd_y)) and (len(bib_x)==len(bib_y)) ):
        plot_2d_histos(signal_x, qcd_x, bib_x, name_x, xmin, xmax, x_bins,  signal_y, qcd_y, bib_y, name_y, ymin, ymax, y_bins, prefix) 



def do_plotting(signal,qcd,bib,name,xmin,xmax,bins, prefix):

    filter_nn_MSeg = [col for col in signal if col.startswith(name)]

    signal_MSeg = remove_values_from_list(signal[filter_nn_MSeg].values.flatten(),np.nan)
    qcd_MSeg = remove_values_from_list(qcd[filter_nn_MSeg].values.flatten(),np.nan)
    bib_MSeg = remove_values_from_list(bib[filter_nn_MSeg].values.flatten(),np.nan)


    plot_three_histos(signal_MSeg,qcd_MSeg,bib_MSeg,name,xmin,xmax,bins, prefix)

def do_plotting_withCut(signal,qcd,bib,name,xmin,xmax,bins, prefix, cutVariable, cutValue):

    filter_nn_MSeg = [col for col in signal if col.startswith(name)]
    filter_nn_cut = [col for col in signal if col.startswith(cutVariable)]

    signal_list_below_cut = []
    qcd_list_below_cut = []
    bib_list_below_cut = []
    signal_list_above_cut = []
    qcd_list_above_cut = []
    bib_list_above_cut = []

    if ( len(filter_nn_MSeg)==len(filter_nn_cut) ):
	    for (original, cutVar)  in zip(filter_nn_MSeg,filter_nn_cut):
                signal_list_below_cut.append( remove_values_from_list((signal[original])[(signal[cutVar] < cutValue) & (signal[cutVar] > -cutValue)].values.flatten(),np.nan) )
                qcd_list_below_cut.append( remove_values_from_list((qcd[original])[(qcd[cutVar] < cutValue) & (qcd[cutVar] > -cutValue)].values.flatten(),np.nan) )
                bib_list_below_cut.append( remove_values_from_list((bib[original])[(bib[cutVar] < cutValue)  & (bib[cutVar] > -cutValue)].values.flatten(),np.nan) )
                signal_list_above_cut.append( remove_values_from_list((signal[original])[(signal[cutVar] > cutValue) | (signal[cutVar] < -cutValue)].values.flatten(),np.nan) )
                qcd_list_above_cut.append( remove_values_from_list((qcd[original])[(qcd[cutVar] > cutValue) | (qcd[cutVar] < -cutValue)].values.flatten(),np.nan) )
                bib_list_above_cut.append( remove_values_from_list((bib[original])[(bib[cutVar] > cutValue) | (bib[cutVar] < -cutValue)].values.flatten(),np.nan) )
    elif ( len(filter_nn_MSeg)>len(filter_nn_cut) ):
	    for (original, cutVar)  in zip(filter_nn_MSeg,cycle(filter_nn_cut)):
                signal_list_below_cut.append( remove_values_from_list((signal[original])[(signal[cutVar] < cutValue) & (signal[cutVar] > -cutValue)].values.flatten(),np.nan) )
                qcd_list_below_cut.append( remove_values_from_list((qcd[original])[(qcd[cutVar] < cutValue) & (qcd[cutVar] > -cutValue)].values.flatten(),np.nan) )
                bib_list_below_cut.append( remove_values_from_list((bib[original])[(bib[cutVar] < cutValue)  & (bib[cutVar] > -cutValue)].values.flatten(),np.nan) )
                signal_list_above_cut.append( remove_values_from_list((signal[original])[(signal[cutVar] > cutValue) | (signal[cutVar] < -cutValue)].values.flatten(),np.nan) )
                qcd_list_above_cut.append( remove_values_from_list((qcd[original])[(qcd[cutVar] > cutValue) | (qcd[cutVar] < -cutValue)].values.flatten(),np.nan) )
                bib_list_above_cut.append( remove_values_from_list((bib[original])[(bib[cutVar] > cutValue) | (bib[cutVar] < -cutValue)].values.flatten(),np.nan) )
    else:
	    for (original, cutVar)  in zip(cycle(filter_nn_MSeg),filter_nn_cut):
                signal_list_below_cut.append( remove_values_from_list((signal[original])[(signal[cutVar] < cutValue) & (signal[cutVar] > -cutValue)].values.flatten(),np.nan) )
                qcd_list_below_cut.append( remove_values_from_list((qcd[original])[(qcd[cutVar] < cutValue) & (qcd[cutVar] > -cutValue)].values.flatten(),np.nan) )
                bib_list_below_cut.append( remove_values_from_list((bib[original])[(bib[cutVar] < cutValue)  & (bib[cutVar] > -cutValue)].values.flatten(),np.nan) )
                signal_list_above_cut.append( remove_values_from_list((signal[original])[(signal[cutVar] > cutValue) | (signal[cutVar] < -cutValue)].values.flatten(),np.nan) )
                qcd_list_above_cut.append( remove_values_from_list((qcd[original])[(qcd[cutVar] > cutValue) | (qcd[cutVar] < -cutValue)].values.flatten(),np.nan) )
                bib_list_above_cut.append( remove_values_from_list((bib[original])[(bib[cutVar] > cutValue) | (bib[cutVar] < -cutValue)].values.flatten(),np.nan) )

        
    '''
    above_cut_signal = signal[signal[cutVariable] > cutValue]
    above_cut_qcd = qcd[qcd[cutVariable] > cutValue]
    above_cut_bib = bib[bib[cutVariable] > cutValue]

    below_cut_signal = signal[signal[cutVariable] < cutValue]
    below_cut_qcd = qcd[qcd[cutVariable] < cutValue]
    below_cut_bib = bib[bib[cutVariable] < cutValue]

    signal_MSeg = remove_values_from_list(signal[filter_nn_MSeg].values.flatten(),np.nan)
    qcd_MSeg = remove_values_from_list(qcd[filter_nn_MSeg].values.flatten(),np.nan)
    bib_MSeg = remove_values_from_list(bib[filter_nn_MSeg].values.flatten(),np.nan)
    '''

    signal_list_below_cut = np.concatenate([np.array(i) for i in signal_list_below_cut])
    qcd_list_below_cut = np.concatenate([np.array(i) for i in qcd_list_below_cut])
    bib_list_below_cut = np.concatenate([np.array(i) for i in bib_list_below_cut])

    signal_list_above_cut = np.concatenate([np.array(i) for i in signal_list_above_cut])
    qcd_list_above_cut = np.concatenate([np.array(i) for i in qcd_list_above_cut])
    bib_list_above_cut = np.concatenate([np.array(i) for i in bib_list_above_cut])

    plot_three_histos_withCut(np.ravel(signal_list_below_cut).flatten(),np.ravel(qcd_list_below_cut).flatten(),np.ravel(bib_list_below_cut).flatten(),name,xmin,xmax,bins, prefix, cutVariable, "<", cutValue)
    plot_three_histos_withCut(np.ravel(signal_list_above_cut).flatten(),np.ravel(qcd_list_above_cut).flatten(),np.ravel(bib_list_above_cut).flatten(),name,xmin,xmax,bins, prefix, cutVariable, ">", cutValue)

def do_truth_plotting(signal,truth_dist,truth_xmin,truth_xmax,truth_bins,prefix):
    for name,xmin,xmax,bins in zip(truth_dist,truth_xmin,truth_xmax,truth_bins):
        plt.hist(signal[name], range=(xmin,xmax),  density=True, color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=bins,label="Signal")
        plt.xlabel(name)
        plt.legend()
        plt.savefig("plots/" + name+ prefix +".png", format='png', transparent=False)
        plt.clf()


def plot_vars(data, prefix=""):

    signal = data[data.label == 1]
    print(signal.shape[0])
    qcd = data[data.label == 0]
    print(qcd.shape[0])
    bib = data[data.label == 2]
    print(bib.shape[0])

    if "cleanJets" in prefix:
        signal = signal[signal.jet_isClean_LooseBadLLP == 1]
        qcd = qcd[qcd.jet_isClean_LooseBadLLP == 1]
        bib = bib[bib.jet_isClean_LooseBadLLP == 1]


    #TODO: BE SMARTER ABOUT THIS!!!!!!
    signal_clus_pt = signal.iloc[:,slice(14,14+28*20,28)]
    qcd_clus_pt = qcd.iloc[:,slice(14,14+28*20,35)]
    bib_clus_pt = bib.iloc[:,slice(14,14+28*20,35)]

    signal_clus_eta = signal.iloc[:,slice(15,15+28*20,28)]
    qcd_clus_eta = qcd.iloc[:,slice(15,15+28*20,28)]
    bib_clus_eta = bib.iloc[:,slice(15,15+28*20,28)]

    signal_clus_phi = signal.iloc[:,slice(16,16+28*20,28)]
    qcd_clus_phi = qcd.iloc[:,slice(16,16+28*20,28)]
    bib_clus_phi = bib.iloc[:,slice(16,16+28*20,28)]

    #jet_eta studies
    signal['aux_llp_jet_eta_difference'] = signal['aux_llp_eta'] - signal['jet_eta']

    signal_all_clus_pt = remove_values_from_list(signal_clus_pt.values.flatten(),np.nan)
    qcd_all_clus_pt = remove_values_from_list(qcd_clus_pt.values.flatten(),np.nan)
    bib_all_clus_pt = remove_values_from_list(bib_clus_pt.values.flatten(),np.nan)

    signal_all_clus_eta = remove_values_from_list(signal_clus_eta.values.flatten(),np.nan)
    qcd_all_clus_eta = remove_values_from_list(qcd_clus_eta.values.flatten(),np.nan)
    bib_all_clus_eta = remove_values_from_list(bib_clus_eta.values.flatten(),np.nan)

    signal_all_clus_phi = remove_values_from_list(signal_clus_phi.values.flatten(),np.nan)
    qcd_all_clus_phi = remove_values_from_list(qcd_clus_phi.values.flatten(),np.nan)
    bib_all_clus_phi = remove_values_from_list(bib_clus_phi.values.flatten(),np.nan)
 
    plot_constit(signal_all_clus_eta,signal_all_clus_phi, signal_all_clus_pt, "signal_constits", prefix)
    plot_constit(bib_all_clus_eta,bib_all_clus_phi, bib_all_clus_pt, "bib_constits", prefix)
    plot_constit(qcd_all_clus_eta,qcd_all_clus_phi, qcd_all_clus_pt, "qcd_constits", prefix)

    xmin_dict = {"jet_pt":0, "jet_eta":-2.5, "jet_phi":-3.14, "jet_isClean_LooseBadLLP":0, "jet_E":0, "clus_pt":0,"clus_eta":-2.5,"clus_phi":-3.14,"e_PreSamplerB":0,"e_EMB1":0,"e_EMB2":0,"e_EMB3":0,"e_PreSamplerE":0,"e_EME1":0,"e_EME2":0,"e_EME3":0,"e_HEC0":0,"e_HEC1":0,"e_HEC2":0,"e_HEC3":0,"e_TileBar0":0,"e_TileBar1":0,"e_TileBar2":0,"e_TileGap1":0,"e_TileGap2":0,"e_TileGap3":0,"e_TileExt0":0,"e_TileExt1":0,"e_TileExt2":0,"e_FCAL0":0,"e_FCAL1":0,"e_FCAL2":0,"clusTime":-10,"nn_track_pt":0,"nn_track_eta":-2.5,"nn_track_phi":-3.14,"nn_track_d0":0,"nn_track_z0":0,"nn_track_PixelShared":-1,"nn_track_PixelSplit":-1,"nn_track_SCTShared":-1,"nn_track_PixelHoles":-1,"nn_track_SCTHoles":-1,"nn_track_PixelHits":-1,"nn_track_SCTHits":-1,"nn_MSeg_etaPos":-4,"nn_MSeg_phiPos":-3.14,"nn_MSeg_etaDir":-8,"nn_MSeg_phiDir":-3.14,"nn_MSeg_t0":-10}

    xmax_dict = {"jet_pt":300000, "jet_eta":2.5, "jet_phi":3.14, "jet_isClean_LooseBadLLP":2, "jet_E":200000, "clus_pt":60000,"clus_eta":2.5,"clus_phi":3.14,"e_PreSamplerB":2000,"e_EMB1":4000,"e_EMB2":6000,"e_EMB3":3000,"e_PreSamplerE":100,"e_EME1":300,"e_EME2":300,"e_EME3":300,"e_HEC0":100,"e_HEC1":100,"e_HEC2":100,"e_HEC3":100,"e_TileBar0":2000,"e_TileBar1":2000,"e_TileBar2":2000,"e_TileGap1":100,"e_TileGap2":100,"e_TileGap3":100,"e_TileExt0":100,"e_TileExt1":100,"e_TileExt2":100,"e_FCAL0":100,"e_FCAL1":100,"e_FCAL2":100,"clusTime":10,"nn_track_pt":10000,"nn_track_eta":2.5,"nn_track_phi":3.14,"nn_track_d0":4,"nn_track_z0":300,"nn_track_PixelShared":10,"nn_track_PixelSplit":10,"nn_track_SCTShared":10,"nn_track_PixelHoles":10,"nn_track_SCTHoles":10,"nn_track_PixelHits":10,"nn_track_SCTHits":10,"nn_MSeg_etaPos":4,"nn_MSeg_phiPos":3.14,"nn_MSeg_etaDir":8,"nn_MSeg_phiDir":3.14,"nn_MSeg_t0":10}

    bin_dict = {"jet_pt":40, "jet_eta":20, "jet_phi":20, "jet_isClean_LooseBadLLP":2, "jet_E":40, "clus_pt":40,"clus_eta":20,"clus_phi":20,"e_PreSamplerB":20,"e_EMB1":20,"e_EMB2":20,"e_EMB3":20,"e_PreSamplerE":20,"e_EME1":20,"e_EME2":20,"e_EME3":20,"e_HEC0":20,"e_HEC1":20,"e_HEC2":20,"e_HEC3":20,"e_TileBar0":20,"e_TileBar1":20,"e_TileBar2":20,"e_TileGap1":20,"e_TileGap2":20,"e_TileGap3":20,"e_TileExt0":20,"e_TileExt1":20,"e_TileExt2":20,"e_FCAL0":20,"e_FCAL1":20,"e_FCAL2":20,"clusTime":20,"nn_track_pt":40,"nn_track_eta":20,"nn_track_phi":20,"nn_track_d0":30,"nn_track_z0":30,"nn_track_PixelShared":11,"nn_track_PixelSplit":11,"nn_track_SCTShared":11,"nn_track_PixelHoles":11,"nn_track_SCTHoles":11,"nn_track_PixelHits":11,"nn_track_SCTHits":11,"nn_MSeg_etaPos":20,"nn_MSeg_phiPos":20,"nn_MSeg_etaDir":20,"nn_MSeg_phiDir":20,"nn_MSeg_t0":20}

    if "processing" in prefix:
        xmin_dict = {"jet_pt":0, "jet_eta":-1, "jet_phi":-1, "jet_isClean_LooseBadLLP":0, "jet_E":0, "clus_pt":0,"clus_eta":-1,"clus_phi":-1,"e_PreSamplerB":0.05,"e_EMB1":0.05,"e_EMB2":0.05,"e_EMB3":0.05,"e_PreSamplerE":0.05,"e_EME1":0.05,"e_EME2":0.05,"e_EME3":0.05,"e_HEC0":0.05,"e_HEC1":0.05,"e_HEC2":0.05,"e_HEC3":0.05,"e_TileBar0":0.05,"e_TileBar1":0.05,"e_TileBar2":0.05,"e_TileGap1":0.05,"e_TileGap2":0.05,"e_TileGap3":0.05,"e_TileExt0":0.05,"e_TileExt1":0.05,"e_TileExt2":0.05,"e_FCAL0":0.05,"e_FCAL1":0.05,"e_FCAL2":0.05,"clusTime":-10,"nn_track_pt":0,"nn_track_eta":-1,"nn_track_phi":-1,"nn_track_d0":0,"nn_track_z0":0,"nn_track_PixelShared":-1,"nn_track_PixelSplit":-1,"nn_track_SCTShared":-1,"nn_track_PixelHoles":-1,"nn_track_SCTHoles":-1,"nn_track_PixelHits":-1,"nn_track_SCTHits":-1,"nn_MSeg_etaPos":-1,"nn_MSeg_phiPos":-1,"nn_MSeg_etaDir":-8,"nn_MSeg_phiDir":-1,"nn_MSeg_t0":-10}

        xmax_dict = {"jet_pt":1, "jet_eta":1, "jet_phi":1, "jet_isClean_LooseBadLLP":2, "jet_E":1, "clus_pt":1,"clus_eta":1,"clus_phi":1,"e_PreSamplerB":1,"e_EMB1":1,"e_EMB2":1,"e_EMB3":1,"e_PreSamplerE":1,"e_EME1":1,"e_EME2":1,"e_EME3":1,"e_HEC0":1,"e_HEC1":1,"e_HEC2":1,"e_HEC3":1,"e_TileBar0":1,"e_TileBar1":1,"e_TileBar2":1,"e_TileGap1":1,"e_TileGap2":1,"e_TileGap3":1,"e_TileExt0":1,"e_TileExt1":1,"e_TileExt2":1,"e_FCAL0":1,"e_FCAL1":1,"e_FCAL2":1,"clusTime":11,"nn_track_pt":1,"nn_track_eta":1,"nn_track_phi":1,"nn_track_d0":4,"nn_track_z0":300,"nn_track_PixelShared":10,"nn_track_PixelSplit":10,"nn_track_SCTShared":10,"nn_track_PixelHoles":10,"nn_track_SCTHoles":10,"nn_track_PixelHits":10,"nn_track_SCTHits":10,"nn_MSeg_etaPos":1,"nn_MSeg_phiPos":1,"nn_MSeg_etaDir":8,"nn_MSeg_phiDir":1,"nn_MSeg_t0":10}

        bin_dict = {"jet_pt":40, "jet_eta":20, "jet_phi":20, "jet_isClean_LooseBadLLP":2, "jet_E":40, "clus_pt":40,"clus_eta":20,"clus_phi":20,"e_PreSamplerB":20,"e_EMB1":20,"e_EMB2":20,"e_EMB3":20,"e_PreSamplerE":20,"e_EME1":20,"e_EME2":20,"e_EME3":20,"e_HEC0":20,"e_HEC1":20,"e_HEC2":20,"e_HEC3":20,"e_TileBar0":20,"e_TileBar1":20,"e_TileBar2":20,"e_TileGap1":20,"e_TileGap2":20,"e_TileGap3":20,"e_TileExt0":20,"e_TileExt1":20,"e_TileExt2":20,"e_FCAL0":20,"e_FCAL1":20,"e_FCAL2":20,"clusTime":20,"nn_track_pt":40,"nn_track_eta":20,"nn_track_phi":20,"nn_track_d0":30,"nn_track_z0":30,"nn_track_PixelShared":11,"nn_track_PixelSplit":11,"nn_track_SCTShared":11,"nn_track_PixelHoles":11,"nn_track_SCTHoles":11,"nn_track_PixelHits":11,"nn_track_SCTHits":11,"nn_MSeg_etaPos":20,"nn_MSeg_phiPos":20,"nn_MSeg_etaDir":20,"nn_MSeg_phiDir":20,"nn_MSeg_t0":20}

    truth_dist = ["aux_llp_Lxy","aux_llp_Lz","aux_llp_pt","aux_llp_eta","aux_llp_phi"]
    truth_xmin = [1000, 1000, 40000, -2.5, -3.14]
    truth_xmax = [4000, 5000, 200000, 2.5, 3.14]
    truth_bins = [20,20,20,20,20]

    do_plotting_2D(signal,qcd,bib,"nn_MSeg_etaDir",-8,8,40,"nn_MSeg_etaPos", -1.5,2.5,40, prefix)
    do_plotting_2D(signal,signal,signal,"aux_llp_eta",-1.5,1.5,40,"aux_llp_Lxy", 1000,4000,60, prefix)
    do_plotting_2D(signal,signal,signal,"aux_llp_jet_eta_difference",-0.4,0.4,40,"aux_llp_eta", -1.5,1.5,60, prefix)
    do_plotting_withCut(signal,qcd,bib,"nn_MSeg_etaDir",-8,8,20,prefix,"jet_eta",1.0)
    do_truth_plotting(signal,truth_dist,truth_xmin,truth_xmax,truth_bins,prefix)
    #do_plotting_withCut(signal,qcd,bib,"nn_MSeg_etaDir",-8,8,20,prefix,"nn_MSeg_etaPos",1.0)
    for key in xmin_dict:
       pass
       do_plotting(signal,qcd,bib,key,xmin_dict[key],xmax_dict[key],bin_dict[key], prefix)
 
 
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

    plot_three_histos( ( (signal[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_constits", 0, 30, 30, prefix)
    plot_three_histos( (  (signal[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_tracks", 0, 20, 20, prefix)
    plot_three_histos( ( (signal[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_MuonSegments", 0, 70, 70, prefix)

    





