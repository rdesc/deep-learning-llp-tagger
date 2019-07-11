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

    #SCALE Cluster Energy Fraction, then unites layers across different eta ranges
    for i in range(0,30):
        layer_one_ecal_list = ["e_PreSamplerB","e_PreSamplerE"]
        layer_two_ecal_list = ["e_EMB1_0","e_EME1_0","e_FCAL0"]
        layer_three_ecal_list = ["e_EMB2_0","e_EME2_0","e_FCAL1"]
        layer_four_ecal_list = ["e_EMB3_0","e_EME3_0","e_FCAL2"]
        layer_one_ecal_list = tuple(layer_one_ecal_list)
        layer_two_ecal_list = tuple(layer_two_ecal_list)
        layer_three_ecal_list = tuple(layer_three_ecal_list)
        layer_four_ecal_list = tuple(layer_four_ecal_list)
	
        layer_one_hcal_list = ["e_HEC0_0"]  
        layer_two_hcal_list = ["e_HEC0_1","e_TileBar0_0","e_TileGap1_0","e_TileExt0_0"]
        layer_three_hcal_list = ["e_HEC0_2","e_TileBar0_1","e_TileGap1_1","e_TileExt0_1"]
        layer_four_hcal_list = ["e_HEC0_3","e_TileBar0_2","e_TileGap1_2","e_TileExt0_2"]
        layer_one_hcal_list = tuple(layer_one_hcal_list)
        layer_two_hcal_list = tuple(layer_two_hcal_list)
        layer_three_hcal_list = tuple(layer_three_hcal_list)
        layer_four_hcal_list = tuple(layer_four_hcal_list)
	
        filter_one_ecal_list = [col for col in data if col.startswith(layer_one_ecal_list) and col.endswith('_'+str(i))]
        print(filter_one_ecal_list)
        filter_two_ecal_list = [col for col in data if col.startswith(layer_two_ecal_list) and col.endswith('_'+str(i))]
        filter_three_ecal_list = [col for col in data if col.startswith(layer_three_ecal_list) and col.endswith('_'+str(i))]
        filter_four_ecal_list = [col for col in data if col.startswith(layer_four_ecal_list) and col.endswith('_'+str(i))]
	
        filter_one_hcal_list = [col for col in data if col.startswith(layer_one_hcal_list) and col.endswith('_'+str(i))]
        filter_two_hcal_list = [col for col in data if col.startswith(layer_two_hcal_list) and col.endswith('_'+str(i))]
        filter_three_hcal_list = [col for col in data if col.startswith(layer_three_hcal_list) and col.endswith('_'+str(i))]
        filter_four_hcal_list = [col for col in data if col.startswith(layer_four_hcal_list) and col.endswith('_'+str(i))]
	
        data['l1_ecal_'+str(i)] = data[filter_one_ecal_list].sum(axis=1)
        data['l2_ecal_'+str(i)] = data[filter_two_ecal_list].sum(axis=1)
        data['l3_ecal_'+str(i)] = data[filter_three_ecal_list].sum(axis=1)
        data['l4_ecal_'+str(i)] = data[filter_four_ecal_list].sum(axis=1)
        
        data['l1_hcal_'+str(i)] = data[filter_one_hcal_list].sum(axis=1)
        data['l2_hcal_'+str(i)] = data[filter_two_hcal_list].sum(axis=1)
        data['l3_hcal_'+str(i)] = data[filter_three_hcal_list].sum(axis=1)
        data['l4_hcal_'+str(i)] = data[filter_four_hcal_list].sum(axis=1)
        
        filter_clus_eFrac = [col for col in data if col.startswith("e_") and col.endswith('_'+str(i))]
        data['sum_eFrac'] = data[filter_clus_eFrac].sum(axis=1)
        
        data['l1_ecal_'+str(i)] = data['l1_ecal_'+str(i)].divide(data['sum_eFrac'], axis='index')
        data['l2_ecal_'+str(i)] = data['l2_ecal_'+str(i)].divide(data['sum_eFrac'], axis='index')
        data['l3_ecal_'+str(i)] = data['l3_ecal_'+str(i)].divide(data['sum_eFrac'], axis='index')
        data['l4_ecal_'+str(i)] = data['l4_ecal_'+str(i)].divide(data['sum_eFrac'], axis='index')
        
        data['l1_hcal_'+str(i)] = data['l1_hcal_'+str(i)].divide(data['sum_eFrac'], axis='index')
        data['l2_hcal_'+str(i)] = data['l2_hcal_'+str(i)].divide(data['sum_eFrac'], axis='index')
        data['l3_hcal_'+str(i)] = data['l3_hcal_'+str(i)].divide(data['sum_eFrac'], axis='index')
        data['l4_hcal_'+str(i)] = data['l4_hcal_'+str(i)].divide(data['sum_eFrac'], axis='index')
	
	#data[filter_clus_eFrac] = data[filter_clus_eFrac].divide(data['sum_eFrac'], axis='index')
	
    layerDelete = [col for col in data if col.startswith("e_")]
    for item in layerDelete:
        del data[item]
    
    del data['sum_eFrac']

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
 
    #Shuffle all jets
    data = data.sample(frac=1).reset_index(drop=True)

    return data

