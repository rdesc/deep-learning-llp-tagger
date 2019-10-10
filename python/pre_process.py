import numpy as np

import seaborn as sns

import pandas as pd

def pre_process(data, min_pt, max_pt):


    #Now For Muon Segments
    print("Pre-processing Muon Segments")

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

    '''
    print("SATART!")

    print(list(data[filter_MSeg_phi].iloc[1,:]))
    print(list(data[filter_MSeg_eta].iloc[1,:]))

    temp_sort = np.absolute(data[filter_MSeg_phi].to_numpy())
    temp_sort = np.around(temp_sort,decimals=6)
    print(temp_sort)
    temp_sort = np.argsort(temp_sort,axis=1, kind='mergesort')

    mseg_names = ['nn_MSeg_etaPos','nn_MSeg_phiPos','nn_MSeg_etaDir','nn_MSeg_phiDir','nn_MSeg_t0']
    for item in mseg_names:
        filter_temp = [col for col in data if col.startswith(item)]
        data_temp = data[filter_temp].to_numpy()
        data_temp = np.array(list(map(lambda x, y: y[x], temp_sort, data_temp)))
        data[filter_temp] = data_temp
    '''

    filter_MSeg_etaDir = [col for col in data if col.startswith("nn_MSeg_etaDir")]
    filter_MSeg_t0 = [col for col in data if col.startswith("nn_MSeg_t0")]

    print(list(data[filter_MSeg_phi].iloc[1,:]))
    print(list(data[filter_MSeg_eta].iloc[1,:]))

    np.set_printoptions(precision=17)


    print("Pre-processing jets")

    print(max_pt)
    print(min_pt)

    #SCALE JET PT
    data["jet_pt"] = data["jet_pt"].sub(float(min_pt), axis='index')
    data["jet_pt"] = data["jet_pt"].divide( (float(max_pt) - float(min_pt)), axis='index')


    print("Pre-processing clusters")
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

    print("Pre-processing cluster energy fraction")

    #SCALE Cluster Energy Fraction, then unites layers across different eta ranges
    for i in range(0,30):
        layer_one_ecal_list = ["e_PreSamplerB","e_PreSamplerE"]
        layer_two_ecal_list = ["e_EMB1","e_EME1","e_FCAL0"]
        layer_three_ecal_list = ["e_EMB2","e_EME2","e_FCAL1"]
        layer_four_ecal_list = ["e_EMB3","e_EME3","e_FCAL2"]
        layer_one_ecal_list = tuple(layer_one_ecal_list)
        layer_two_ecal_list = tuple(layer_two_ecal_list)
        layer_three_ecal_list = tuple(layer_three_ecal_list)
        layer_four_ecal_list = tuple(layer_four_ecal_list)
	
        layer_one_hcal_list = ["e_HEC0"]  
        layer_two_hcal_list = ["e_HEC1","e_TileBar0","e_TileGap1","e_TileExt0"]
        layer_three_hcal_list = ["e_HEC2","e_TileBar1","e_TileGap2","e_TileExt1"]
        layer_four_hcal_list = ["e_HEC3","e_TileBar2","e_TileGap3","e_TileExt2"]
        layer_one_hcal_list = tuple(layer_one_hcal_list)
        layer_two_hcal_list = tuple(layer_two_hcal_list)
        layer_three_hcal_list = tuple(layer_three_hcal_list)
        layer_four_hcal_list = tuple(layer_four_hcal_list)
	
        filter_one_ecal_list = [col for col in data if col.startswith(layer_one_ecal_list) and col.endswith('_'+str(i))]
        print(filter_one_ecal_list)
        filter_two_ecal_list = [col for col in data if col.startswith(layer_two_ecal_list) and col.endswith('_'+str(i))]
        print(filter_two_ecal_list)
        filter_three_ecal_list = [col for col in data if col.startswith(layer_three_ecal_list) and col.endswith('_'+str(i))]
        print(filter_three_ecal_list)
        filter_four_ecal_list = [col for col in data if col.startswith(layer_four_ecal_list) and col.endswith('_'+str(i))]
        print(filter_four_ecal_list)
	
        filter_one_hcal_list = [col for col in data if col.startswith(layer_one_hcal_list) and col.endswith('_'+str(i))]
        print(filter_one_hcal_list)
        filter_two_hcal_list = [col for col in data if col.startswith(layer_two_hcal_list) and col.endswith('_'+str(i))]
        print(filter_two_hcal_list)
        filter_three_hcal_list = [col for col in data if col.startswith(layer_three_hcal_list) and col.endswith('_'+str(i))]
        print(filter_three_hcal_list)
        filter_four_hcal_list = [col for col in data if col.startswith(layer_four_hcal_list) and col.endswith('_'+str(i))]
        print(filter_four_hcal_list)
	
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
    #for item in layerDelete:
    #del data[item]
    data = data.drop(layerDelete, axis=1)
    
    del data['sum_eFrac']

    filter_clus_l1ecal = [col for col in data if col.startswith("l1_ecal")]
    filter_clus_l2ecal = [col for col in data if col.startswith("l2_ecal")]
    filter_clus_l3ecal = [col for col in data if col.startswith("l3_ecal")]
    filter_clus_l4ecal = [col for col in data if col.startswith("l4_ecal")]
    filter_clus_l1hcal = [col for col in data if col.startswith("l1_hcal")]
    filter_clus_l2hcal = [col for col in data if col.startswith("l2_hcal")]
    filter_clus_l3hcal = [col for col in data if col.startswith("l3_hcal")]
    filter_clus_l4hcal = [col for col in data if col.startswith("l4_hcal")]
    filter_clus_time = [col for col in data if col.startswith("clusTime")]

    '''
    print(list((data[filter_clus_pt].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_eta].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_phi].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_l1ecal].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_l2ecal].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_l3ecal].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_l4ecal].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_l1hcal].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_l2hcal].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_l3hcal].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_l4hcal].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_clus_time].loc[data['eventNumber'] == 15641]).values))
    '''

    #Now For Tracks
    print("Pre-processing tracks")

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

   
    
    #SCALE Track PT
    data[filter_track_pt] = data[filter_track_pt].sub(min_pt, axis='index')
    data[filter_track_pt] = data[filter_track_pt].divide( (max_pt - min_pt), axis='index')

    #Add all etas weighted by pt, then make column that is 1 if positive, -1 if negative
    data['track_sign'] = np.sum(np.multiply(data[filter_track_eta].fillna(0).to_numpy(),data[filter_track_pt].fillna(0).to_numpy()), axis=1)
    data['track_sign'] = data['track_sign'].apply(lambda x: 1*(x>=0) + (-1)*(x<0) )

    #Flip (multiply by -1) according to previously calculated column
    data[filter_track_eta] = data[filter_track_eta].multiply(data["track_sign"], axis='index')

    filter_track_d0 = [col for col in data if col.startswith("nn_track_d0")]
    filter_track_z0 = [col for col in data if col.startswith("nn_track_z0")]
    filter_track_pixelShared = [col for col in data if col.startswith("nn_track_PixelShared")]
    filter_track_pixelSplit = [col for col in data if col.startswith("nn_track_PixelSplit")]
    filter_track_SCTShared = [col for col in data if col.startswith("nn_track_SCTShared")]
    filter_track_PixelHoles = [col for col in data if col.startswith("nn_track_PixelHoles")]
    filter_track_SCTHoles = [col for col in data if col.startswith("nn_track_SCTHoles")]
    filter_track_PixelHits = [col for col in data if col.startswith("nn_track_PixelHits")]
    filter_track_SCTHits = [col for col in data if col.startswith("nn_track_SCTHits")]

    '''
    print(list((data[filter_track_pt].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_eta].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_phi].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_d0].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_z0].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_pixelShared].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_pixelSplit].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_SCTShared].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_PixelHoles].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_SCTHoles].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_PixelHits].loc[data['eventNumber'] == 15641]).values))
    print(list((data[filter_track_SCTHits].loc[data['eventNumber'] == 15641]).values))
    '''

 
    #Shuffle all jets
    print("Shuffling pre-processed data")
    data = data.sample(frac=1).reset_index(drop=True)

    return data

