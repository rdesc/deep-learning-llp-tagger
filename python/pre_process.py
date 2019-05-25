import numpy as np
import seaborn as sns

import pandas as pd

def pre_process(data, min_pt, max_pt):


    #SCALE JET PT
    data["jet_pt"] = data["jet_pt"].sub(min_pt, axis='index')
    data["jet_pt"] = data["jet_pt"].divide( (max_pt - min_pt), axis='index')


    #DO PHI, ETA Shift

    #Get all eta columns
    filter_clus_eta = [col for col in data if col.startswith("clus_eta")]
    #Get all phi columns
    filter_clus_phi = [col for col in data if col.startswith("clus_phi")]
    #Get all pt columns
    filter_clus_pt = [col for col in data if col.startswith("clus_pt")]

    #Subtract the eta of first cluster(largest pt) from all other
    data[filter_clus_eta] = data[filter_clus_eta].sub(data["clus_eta_0"], axis='index')

    #Subtract the phi of first cluster(largest pt) from all other
    data[filter_clus_phi] = data[filter_clus_phi].sub(data["clus_phi_0"], axis='index')



    #Do eta, phi FLIP

    #Add all etas weighted by pt, then make column that is 1 if positive, -1 if negative
    data['clus_sign'] = np.sum(np.multiply(data[filter_clus_eta].fillna(0).to_numpy(),data[filter_clus_pt].fillna(0).to_numpy()), axis=1)
    data['clus_sign'] = data['clus_sign'].apply(lambda x: 1*(x>=0) + (-1)*(x<0) )

    #Flip (multiply by -1) according to previously calculated column
    data[filter_clus_eta] = data[filter_clus_eta].multiply(data["clus_sign"], axis='index')
   
    
    #SCALE CLUSTER PT
    data[filter_clus_pt] = data[filter_clus_pt].sub(min_pt, axis='index')
    data[filter_clus_pt] = data[filter_clus_pt].divide( (max_pt - min_pt), axis='index')

    #SCALE Cluster Energy Fraction

    filter_clus_eFrac = [col for col in data if col.startswith("e_")]
    data['sum_eFrac'] = data[filter_clus_eFrac].sum(axis=1)
    data[filter_clus_eFrac] = data[filter_clus_eFrac].divide(data['sum_eFrac'], axis='index')

    #Now For Tracks

    #Get all eta columns
    filter_track_eta = [col for col in data if col.startswith("nn_track_eta")]
    #Get all phi columns
    filter_track_phi = [col for col in data if col.startswith("nn_track_phi")]
    #Get all pt columns
    filter_track_pt = [col for col in data if col.startswith("nn_track_pt")]

    #Subtract the eta of the jet from all tracks
    data[filter_track_eta] = data[filter_track_eta].sub(data["jet_eta"], axis='index')

    #Subtract the phi of the jet from all tracks
    data[filter_track_phi] = data[filter_track_phi].sub(data["jet_phi"], axis='index')


    #Do eta, phi FLIP

    #Add all etas weighted by pt, then make column that is 1 if positive, -1 if negative
    data['track_sign'] = np.sum(np.multiply(data[filter_track_eta].fillna(0).to_numpy(),data[filter_track_pt].fillna(0).to_numpy()), axis=1)
    data['track_sign'] = data['track_sign'].apply(lambda x: 1*(x>=0) + (-1)*(x<0) )

    #Flip (multiply by -1) according to previously calculated column
    data[filter_track_eta] = data[filter_track_eta].multiply(data["track_sign"], axis='index')
   
    
    #SCALE Track PT
    data[filter_track_pt] = data[filter_track_pt].sub(min_pt, axis='index')
    data[filter_track_pt] = data[filter_track_pt].divide( (max_pt - min_pt), axis='index')


    #Now For Muon Segments

    #Get all eta Position columns
    filter_MSeg_eta = [col for col in data if col.startswith("nn_MSeg_etaPos")]
    #Get all phi Position columns
    filter_MSeg_phi = [col for col in data if col.startswith("nn_MSeg_phiPos")]
    #Get all phi Direction  columns
    filter_MSeg_phiDir = [col for col in data if col.startswith("nn_MSeg_phiDir")]

    #Subtract the eta of the jet from all MSegs
    data[filter_MSeg_eta] = data[filter_MSeg_eta].sub(data["jet_eta"], axis='index')

    #Subtract the phi of the jet from all MSegs
    data[filter_MSeg_phi] = data[filter_MSeg_phi].sub(data["jet_phi"], axis='index')

    #Subtract the phi of the jet from all MSegs Dir
    data[filter_MSeg_phiDir] = data[filter_MSeg_phiDir].sub(data["jet_phi"], axis='index')


    return data

