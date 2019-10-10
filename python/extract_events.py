import numpy as np
import seaborn as sns

import pandas as pd

import uproot

import concurrent.futures
import multiprocessing

import itertools

import sys

from glob import glob

import time
import re

def load_root(filename):
    uproot.open(filename).keys()

def load_data(filename):
    print("loading " + filename)
    x = np.load(filename, encoding='latin1')
    x_data = x['data']
    #x_data = x_data[0:1000,:]
    return x_data

def find_number(text, c):
    return re.findall(r'%s(\d+)' % c, text)


def process_qcd_events(array):

    ###Added while waiting for QCD samples###
    #TODO: delete when those come

    '''
    array['test'] = array.apply(lambda x: ([((x['jet_eta'] - i)**2+(x['jet_phi'] - j)**2) for i,j in zip(x['LLP_eta'],x['LLP_phi'])]) , axis=1)
    #array['test_0'] = (array.test[array['test'].apply(lambda x: len(x[0])) > 0]).apply(lambda x: np.argmin((x[0])*(1+(-2)*(min(x[0]) > 0.04))))
    #test_0: LLP 0, test_1: LLP_1: only choose lowest DR, which is less than 0.4, make negative if higher
    array['test_0'] = (array.test[array['test'].apply(lambda x: len(x[0])) > 0]).apply(lambda x: np.argmin((x[0]))*(1+(-2)*(min(x[0])>0.4)*1))
    array['test_1'] = (array.test[array['test'].apply(lambda x: len(x[1])) > 0]).apply(lambda x: np.argmin((x[1]))*(1+(-2)*(min(x[1])>0.4)*1))
    #array['test_1'] = (array.test[array['test'].apply(lambda x: len(x[1])) > 0]).apply(lambda x: np.argmin((x[1])*(1+(-2)*(min(x[1]) > 0.04))))
    #TODO Look for ways to still see if two available jets when min is same jet
    #Choose only non-negative indices, and where it's not the same jet matched to both LLPs
    array['test_0'] = array.test_0[array.test_0 >=0]
    array['test_1'] = array.test_1[array.test_1 >=0]
    array['test_1'] = array['test_1'].loc[~(array['test_0'] == array['test_1'])]

    array['test'] = array.apply(lambda x: x['jet_index'] != x['test_0'], axis=1)
    array['jet_pt'] = array.apply(lambda x: x['jet_pt'][x['test'] == True], axis=1)
    array['jet_isClean_LooseBadLLP'] = array.apply(lambda x: x['jet_isClean_LooseBadLLP'][x['test'] == True], axis=1)
    array['jet_eta'] = array.apply(lambda x: x['jet_eta'][x['test'] == True], axis=1)
    array['jet_phi'] = array.apply(lambda x: x['jet_phi'][x['test'] == True], axis=1)
    array['jet_index'] = array.apply(lambda x: x['jet_index'][x['test'] == True], axis=1)

    array['test'] = array.apply(lambda x: x['jet_index'] != x['test_1'], axis=1)
    array['jet_pt'] = array.apply(lambda x: x['jet_pt'][x['test'] == True], axis=1)
    array['jet_isClean_LooseBadLLP'] = array.apply(lambda x: x['jet_isClean_LooseBadLLP'][x['test'] == True], axis=1)
    array['jet_eta'] = array.apply(lambda x: x['jet_eta'][x['test'] == True], axis=1)
    array['jet_phi'] = array.apply(lambda x: x['jet_phi'][x['test'] == True], axis=1)
    array['jet_index'] = array.apply(lambda x: x['jet_index'][x['test'] == True], axis=1)

    ###End of added for lack of QCD samples###
    '''

    array['test'] = array.apply(lambda x: x['jet_pt'] > 40000, axis=1)
    array['jet_pt'] = array.apply(lambda x: x['jet_pt'][x['test'] == True], axis=1)
    array['jet_isClean_LooseBadLLP'] = array.apply(lambda x: x['jet_isClean_LooseBadLLP'][x['test'] == True], axis=1)
    array['jet_eta'] = array.apply(lambda x: x['jet_eta'][x['test'] == True], axis=1)
    array['jet_phi'] = array.apply(lambda x: x['jet_phi'][x['test'] == True], axis=1)
    array['jet_index'] = array.apply(lambda x: x['jet_index'][x['test'] == True], axis=1)

    array['test'] = array.apply(lambda x: x['jet_pt'] < 500000, axis=1)
    array['jet_pt'] = array.apply(lambda x: x['jet_pt'][x['test'] == True], axis=1)
    array['jet_isClean_LooseBadLLP'] = array.apply(lambda x: x['jet_isClean_LooseBadLLP'][x['test'] == True], axis=1)
    array['jet_eta'] = array.apply(lambda x: x['jet_eta'][x['test'] == True], axis=1)
    array['jet_phi'] = array.apply(lambda x: x['jet_phi'][x['test'] == True], axis=1)
    array['jet_index'] = array.apply(lambda x: x['jet_index'][x['test'] == True], axis=1)

    array['test'] = array.apply(lambda x: x['jet_eta'] > -2.5, axis=1)
    array['jet_pt'] = array.apply(lambda x: x['jet_pt'][x['test'] == True], axis=1)
    array['jet_isClean_LooseBadLLP'] = array.apply(lambda x: x['jet_isClean_LooseBadLLP'][x['test'] == True], axis=1)
    array['jet_eta'] = array.apply(lambda x: x['jet_eta'][x['test'] == True], axis=1)
    array['jet_phi'] = array.apply(lambda x: x['jet_phi'][x['test'] == True], axis=1)
    array['jet_index'] = array.apply(lambda x: x['jet_index'][x['test'] == True], axis=1)

    array['test'] = array.apply(lambda x: x['jet_eta'] < 2.5, axis=1)
    array['jet_pt'] = array.apply(lambda x: x['jet_pt'][x['test'] == True], axis=1)
    array['jet_isClean_LooseBadLLP'] = array.apply(lambda x: x['jet_isClean_LooseBadLLP'][x['test'] == True], axis=1)
    array['jet_eta'] = array.apply(lambda x: x['jet_eta'][x['test'] == True], axis=1)
    array['jet_phi'] = array.apply(lambda x: x['jet_phi'][x['test'] == True], axis=1)
    array['jet_index'] = array.apply(lambda x: x['jet_index'][x['test'] == True], axis=1)

    array_0 = array.loc[array.jet_index.apply(lambda x: len(x) > 0)].copy()
    array_1 = array.loc[array.jet_index.apply(lambda x: len(x) > 1)].copy()

 
    array_0['jet_pt'] = array_0['jet_pt'].apply(lambda x: x[0])
    array_0['jet_isClean_LooseBadLLP'] = array_0['jet_isClean_LooseBadLLP'].apply(lambda x: x[0])
    array_0['jet_eta'] = array_0['jet_eta'].apply(lambda x: x[0])
    array_0['jet_phi'] = array_0['jet_phi'].apply(lambda x: x[0])
    array_0['jet_index'] = array_0['jet_index'].apply(lambda x: x[0])

    array_1['jet_pt'] = array_1['jet_pt'].apply(lambda x: x[1])
    array_1['jet_isClean_LooseBadLLP'] = array_1['jet_isClean_LooseBadLLP'].apply(lambda x: x[1])
    array_1['jet_eta'] = array_1['jet_eta'].apply(lambda x: x[1])
    array_1['jet_phi'] = array_1['jet_phi'].apply(lambda x: x[1])
    array_1['jet_index'] = array_1['jet_index'].apply(lambda x: x[1])

    num_cluster_variables = len((array_0.loc[:,'clus_pt':'clusTime']).columns.values)
    num_track_variables = len((array_0.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values)
    num_muon_variables = len((array_0.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values)

    num_max_constits = 30
    num_max_tracks = 20
    num_max_muonSegs = 70

    size_0 = array_0.shape[0] + array_1.shape[0]
    size_1 = (num_cluster_variables*num_max_constits) + (num_track_variables*num_max_tracks) + (num_muon_variables*num_max_muonSegs) + 14
    x_data = np.full([size_0,size_1],np.nan, dtype='float32')
    

    x_data[0:array_0.shape[0],0] = np.zeros(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],0] = np.zeros(array_1.shape[0])

    x_data[0:array_0.shape[0],1] = np.array([*array_0['mcEventWeight'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],1] = np.array([*array_1['mcEventWeight'].to_numpy()])

    x_data[0:array_0.shape[0],2] = np.ones(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],2] = np.ones(array_1.shape[0])

    x_data[0:array_0.shape[0],3] = np.array([*array_0['jet_pt'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],3] = np.array([*array_1['jet_pt'].to_numpy()])

    x_data[0:array_0.shape[0],4] = np.array([*array_0['jet_eta'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],4] = np.array([*array_1['jet_eta'].to_numpy()])

    x_data[0:array_0.shape[0],5] = np.array([*array_0['jet_phi'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],5] = np.array([*array_1['jet_phi'].to_numpy()])

    x_data[0:array_0.shape[0],6] = np.array([*array_0['jet_isClean_LooseBadLLP'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],6] = np.array([*array_1['jet_isClean_LooseBadLLP'].to_numpy()])

    x_data[0:array_0.shape[0],7] = np.zeros(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],7] = np.zeros(array_1.shape[0])
  
    x_data[0:array_0.shape[0],8] = np.zeros(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],8] = np.zeros(array_1.shape[0])

    x_data[0:array_0.shape[0],9] = np.zeros(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],9] = np.zeros(array_1.shape[0])

    x_data[0:array_0.shape[0],10] = np.zeros(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],10] = np.zeros(array_1.shape[0])

    x_data[0:array_0.shape[0],11] = np.zeros(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],11] = np.zeros(array_1.shape[0])

    x_data[0:array_0.shape[0],12] = np.zeros(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],12] = np.zeros(array_1.shape[0])

    x_data[0:array_0.shape[0],13] = np.zeros(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],13] = np.zeros(array_1.shape[0])
  

    clus_sort_index_0 = np.zeros(num_max_constits)
    track_sort_index_0 = np.zeros(num_max_constits)
    clus_sort_index_1 = np.zeros(num_max_tracks)
    track_sort_index_1 = np.zeros(num_max_tracks)


    #print(array_0.clus_pt.apply(lambda x: len(x)))
    counter_cluster=14
    for item in (array_0.loc[:,'clus_pt':'clusTime']).columns.values:
        array_0[item] = (array_0.apply(lambda x: x[item][x['cluster_jetIndex'] == int(x['jet_index'])], axis=1))
        #SORRY ABOUT THIS IT IS BAD CODE
        array_0[item] = (array_0[item].apply(lambda x: np.multiply(np.resize(x,num_max_constits),np.concatenate([np.ones(len(x)*(len(x) <= num_max_constits) + num_max_constits*(len(x) > num_max_constits)),np.full((num_max_constits-len(x))*(len(x) < num_max_constits),np.nan, dtype='float32')]) )) ) 
        if item == "clus_pt":
            array_0_np = np.array([*array_0[item].to_numpy()])
            array_0_pt = np.array([*array_0[item].to_numpy()])
            clus_sort_index_0 = np.argsort(array_0_np)
        axis = 1
        index = list(np.ix_(*[np.arange(clus_sort_index_0) for clus_sort_index_0 in array_0_np.shape]))
        array_0_np = np.array([*array_0[item].to_numpy()])
        index[axis] = (-array_0_pt).argsort(axis)
        x_data[0:array_0.shape[0],slice(counter_cluster,counter_cluster+((num_max_constits-1)*num_cluster_variables)+1,num_cluster_variables)] = array_0_np[tuple(index)]
        #print(array_0[item].to_numpy().shape)
        #print(x_data[:,slice(0,112,28)].shape)
        #print((array_0[item].to_numpy()).shape)
        #test[0:array_0.shape[0],0:20] = array_0[item].to_numpy()
        #test = np.asarray(array_0[item].to_numpy()).copy()


        #print( np.pad(array_0[item].to_numpy(),(20),mode='constant', constant_values=0).shape )
        #x_data[:,slice(0,28*20,28)] = array_0[item].to_numpy()
        #print(array_0[item])
        #array_1[item] = (array_1.apply(lambda x: x[item][x['cluster_jetIndex'] == int(x['test_1'])], axis=1))
        array_1[item] = (array_1.apply(lambda x: x[item][x['cluster_jetIndex'] == int(x['jet_index'])], axis=1))
        #SORRY ABOUT THIS IT IS BAD CODE
        array_1[item] = (array_1[item].apply(lambda x: np.multiply(np.resize(x,num_max_constits),np.concatenate([np.ones(len(x)*(len(x) <= num_max_constits) + num_max_constits*(len(x) > num_max_constits)),np.full((num_max_constits-len(x))*(len(x) < num_max_constits),np.nan, dtype='float32')]) )) ) 
        if item == "clus_pt":
            array_1_np = np.array([*array_1[item].to_numpy()])
            array_1_pt = np.array([*array_1[item].to_numpy()])
            clus_sort_index_1 = np.argsort(array_1_np)
        axis = 1
        index = list(np.ix_(*[np.arange(clus_sort_index_1) for clus_sort_index_1 in array_1_np.shape]))
        array_1_np = np.array([*array_1[item].to_numpy()])
        index[axis] = (-array_1_pt).argsort(axis)
        x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],slice(counter_cluster,counter_cluster+((num_max_constits-1)*num_cluster_variables)+1,num_cluster_variables)] = array_1_np[tuple(index)]
        counter_cluster = counter_cluster+1

    counter_tracks=0
    max_counter_cluster = counter_cluster+((num_max_constits-1)*num_cluster_variables)

    for item in (array_0.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values:
        array_0[item] = (array_0.apply(lambda x: x[item][x['nn_track_jetIndex'] == int(x['jet_index'])], axis=1))

        array_0[item] = (array_0[item].apply(lambda x: np.multiply(np.resize(x,num_max_tracks),np.concatenate([np.ones(len(x)*(len(x) <= num_max_tracks) + num_max_tracks*(len(x) > num_max_tracks)),np.full((num_max_tracks-len(x))*(len(x) < num_max_tracks),np.nan, dtype='float32')]) )) ) 
        if item == "nn_track_pt":
            array_0_np = np.array([*array_0[item].to_numpy()])
            array_0_pt = np.array([*array_0[item].to_numpy()])
            track_sort_index_0 = np.argsort(array_0_np)
        axis = 1
        index = list(np.ix_(*[np.arange(track_sort_index_0) for track_sort_index_0 in array_0_np.shape]))
        array_0_np = np.array([*array_0[item].to_numpy()])
        index[axis] = (-array_0_pt).argsort(axis)
        x_data[0:array_0.shape[0],slice(counter_tracks+max_counter_cluster,max_counter_cluster+counter_tracks+((num_max_tracks-1)*num_track_variables)+1,num_track_variables)] = array_0_np[tuple(index)]

        array_1[item] = (array_1.apply(lambda x: x[item][x['nn_track_jetIndex'] == int(x['jet_index'])], axis=1))

        array_1[item] = (array_1[item].apply(lambda x: np.multiply(np.resize(x,num_max_tracks),np.concatenate([np.ones(len(x)*(len(x) <= num_max_tracks) + num_max_tracks*(len(x) > num_max_tracks)),np.full((num_max_tracks-len(x))*(len(x) < num_max_tracks),np.nan, dtype='float32')]) )) ) 
        if item == "nn_track_pt":
            array_1_np = np.array([*array_1[item].to_numpy()])
            array_1_pt = np.array([*array_1[item].to_numpy()])
            track_sort_index_1 = np.argsort(array_1_np)
        axis = 1
        index = list(np.ix_(*[np.arange(track_sort_index_1) for track_sort_index_1 in array_1_np.shape]))
        array_1_np = np.array([*array_1[item].to_numpy()])
        index[axis] = (-array_1_pt).argsort(axis)
        x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],slice(counter_tracks++max_counter_cluster,max_counter_cluster+counter_tracks+((num_max_tracks-1)*num_track_variables)+1,num_track_variables)] = array_1_np[tuple(index)]

        counter_tracks = counter_tracks + 1
     
    counter_muons = 0
    max_counter_tracks =  max_counter_cluster+counter_tracks+((num_max_tracks-1)*num_track_variables)

    for item in (array_0.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values:
        array_0[item] = (array_0.apply(lambda x: x[item][x['nn_MSeg_jetIndex'] == int(x['jet_index'])], axis=1))
        array_0[item] = (array_0[item].apply(lambda x: np.multiply(np.resize(x,num_max_muonSegs),np.concatenate([np.ones(len(x)*(len(x) <= num_max_muonSegs) + num_max_muonSegs*(len(x) > num_max_muonSegs)),np.full((num_max_muonSegs-len(x))*(len(x) < num_max_muonSegs),np.nan, dtype='float32')]) )) )
        array_0_np = np.array([*array_0[item].to_numpy()])
        x_data[0:array_0.shape[0],slice(counter_muons+max_counter_tracks,max_counter_tracks+counter_muons+((num_max_muonSegs-1)*num_muon_variables)+1,num_muon_variables)] = array_0_np

        array_1[item] = (array_1.apply(lambda x: x[item][x['nn_MSeg_jetIndex'] == int(x['jet_index'])], axis=1))

        array_1[item] = (array_1[item].apply(lambda x: np.multiply(np.resize(x,num_max_muonSegs),np.concatenate([np.ones(len(x)*(len(x) <= num_max_muonSegs) + num_max_muonSegs*(len(x) > num_max_muonSegs)),np.full((num_max_muonSegs-len(x))*(len(x) < num_max_muonSegs),np.nan, dtype='float32')]) )) ) 
        array_1_np = np.array([*array_1[item].to_numpy()])
        x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],slice(counter_muons+max_counter_tracks,counter_muons+max_counter_tracks+((num_max_muonSegs-1)*num_muon_variables)+1,num_muon_variables)] = array_1_np

        counter_muons = counter_muons + 1




    #print( (array_0.loc[:,'clus_pt':'clusTime']).columns.values + (array_0.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values + (array_0.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values ) 
    #print( (array_0.loc[:,'clus_pt':'clusTime']).columns.values)

    initial_names = ['label','mcEventWeight','flatWeight','jet_pt','jet_eta','jet_phi','jet_isClean_LooseBadLLP', 'aux_llp_Lxy','aux_llp_Lz','aux_llp_pt','aux_llp_eta','aux_llp_phi','llp_mH','llp_mS']

    constit_cols = ((array_0.loc[:,'clus_pt':'clusTime']).columns.values)
    constit_names = []
    for a in range(num_max_constits):
        for b in constit_cols:
            constit_names.append('%s_%d' % (b, a))

    track_cols = ((array_0.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values)
    track_names = []
    for a in range(num_max_tracks):
        for b in track_cols:
            track_names.append('%s_%d' % (b, a))

    MSeg_cols = ((array_0.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values)
    MSeg_names = []
    for a in range(num_max_muonSegs):
        for b in MSeg_cols:
            MSeg_names.append('%s_%d' % (b, a))

    total_names = initial_names + constit_names + track_names + MSeg_names

    final_dataFrame = pd.DataFrame(x_data, columns = total_names)
  
    return(final_dataFrame)


def process_bib_events(array):
    np.set_printoptions(threshold=sys.maxsize)
    
    array = array.loc[array.HLT_jet_isBIB.apply(lambda x: len(x) == 1)].copy()


    #gives two arrays in 1, [0] corresponds to first LLP, [1] correspodns to second LLP
    array = array[ array.HLT_jet_isBIB == 1].copy()
    print(array.shape[0])
    #test_0: LLP 0, test_1: LLP_1: only choose lowest DR, which is less than 0.4, make negative if higher
    array['test'] = array.apply(lambda x: ([((x['jet_eta'] - i)**2+(x['jet_phi'] - j)**2) for i,j in zip(x['HLT_jet_eta'],x['HLT_jet_phi'])]) , axis=1)
    #TODO check if only one HLT jet each time
    array['test_0'] = (array.test[array['test'].apply(lambda x: len(x[0])) > 0]).apply(lambda x: np.argmin((x[0]))*(1+(-2)*(min(x[0])>0.4)*1))
    #TODO Look for ways to still see if two available jets when min is same jet
    #Choose only non-negative indices, and where it's not the same jet matched to both LLPs
    array = array.loc[array.test_0 >=0]

    array = array.loc[ ((array.test_0 >= 0))].copy() 


    array['jet_pt_hlt']  = array.apply(lambda x: ([(x['jet_pt'][(int(x['test_0']))]) ]) , axis=1)  
    array['jet_eta_hlt']  = (array.apply(lambda x: ([(x['jet_eta'][(int(x['test_0']))]) ]) , axis=1)  )
    array['jet_phi_hlt']  = (array.apply(lambda x: ([(x['jet_phi'][(int(x['test_0']))]) ]) , axis=1)  )
    array['jet_isClean_LooseBadLLP_hlt']  = (array.apply(lambda x: ([(x['jet_isClean_LooseBadLLP'][(int(x['test_0']))]) ]) , axis=1)  )

    array['jet_pt_hlt'] = array.jet_pt_hlt.apply(lambda x: x[0]) 
    array['jet_eta_hlt'] = array.jet_eta_hlt.apply(lambda x: x[0]) 
    array['jet_phi_hlt'] = array.jet_phi_hlt.apply(lambda x: x[0]) 
    array['jet_isClean_LooseBadLLP_hlt'] = array.jet_isClean_LooseBadLLP.apply(lambda x: x[0]) 


    array = array.loc[ array.jet_pt_hlt >= 40000]
    array = array.loc[ array.jet_pt_hlt < 500000]
    array = array.loc[ array.jet_eta_hlt >= -2.5]
    array = array.loc[ array.jet_eta_hlt <= 2.5]


    num_cluster_variables = len((array.loc[:,'clus_pt':'clusTime']).columns.values)
    num_track_variables = len((array.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values)
    num_muon_variables = len((array.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values)

    num_max_constits = 30
    num_max_tracks = 20
    num_max_muonSegs = 70

    size_0 = array.shape[0] 
    size_1 = (num_cluster_variables*num_max_constits) + (num_track_variables*num_max_tracks) + (num_muon_variables*num_max_muonSegs) + 14
    x_data = np.full([size_0,size_1],np.nan, dtype='float32')
    

    x_data[0:array.shape[0],0] = np.ones(array.shape[0])*2

    x_data[0:array.shape[0],1] = np.ones(array.shape[0])

    x_data[0:array.shape[0],2] = np.ones(array.shape[0])

    x_data[0:array.shape[0],3] = np.array([*array['jet_pt_hlt'].to_numpy()])

    x_data[0:array.shape[0],4] = np.array([*array['jet_eta_hlt'].to_numpy()])

    x_data[0:array.shape[0],5] = np.array([*array['jet_phi_hlt'].to_numpy()])

    x_data[0:array.shape[0],6] = np.array([*array['jet_isClean_LooseBadLLP_hlt'].to_numpy()])

    x_data[0:array.shape[0],7] = np.zeros(array.shape[0])
 
    x_data[0:array.shape[0],8] = np.zeros(array.shape[0])

    x_data[0:array.shape[0],9] = np.zeros(array.shape[0])

    x_data[0:array.shape[0],10] = np.zeros(array.shape[0])

    x_data[0:array.shape[0],11] = np.zeros(array.shape[0])

    x_data[0:array.shape[0],12] = np.zeros(array.shape[0])

    x_data[0:array.shape[0],13] = np.zeros(array.shape[0])

    clus_sort_index = np.zeros(num_max_constits)
    track_sort_index = np.zeros(num_max_constits)

  
    #print(array.clus_pt.apply(lambda x: len(x)))
    counter_cluster=14
    for item in (array.loc[:,'clus_pt':'clusTime']).columns.values:
        array[item] = (array.apply(lambda x: x[item][x['cluster_jetIndex'] == int(x['test_0'])], axis=1))
        #SORRY ABOUT THIS IT IS BAD CODE
        array[item] = (array[item].apply(lambda x: np.multiply(np.resize(x,num_max_constits),np.concatenate([np.ones(len(x)*(len(x) <= num_max_constits) + num_max_constits*(len(x) > num_max_constits)),np.full((num_max_constits-len(x))*(len(x) < num_max_constits),np.nan, dtype='float32')]) )) ) 
        if item == "clus_pt":
            array_np = np.array([*array[item].to_numpy()])
            array_pt = np.array([*array[item].to_numpy()])
            clus_sort_index = np.argsort(array_np)
        axis = 1
        index = list(np.ix_(*[np.arange(clus_sort_index) for clus_sort_index in array_np.shape]))
        array_np = np.array([*array[item].to_numpy()])
        index[axis] = (-array_pt).argsort(axis)
        x_data[0:array.shape[0],slice(counter_cluster,counter_cluster+((num_max_constits-1)*num_cluster_variables)+1,num_cluster_variables)] = array_np[tuple(index)]
        counter_cluster = counter_cluster+1

    counter_tracks=0
    max_counter_cluster = counter_cluster+((num_max_constits-1)*num_cluster_variables)

    for item in (array.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values:
        array[item] = (array.apply(lambda x: x[item][x['nn_track_jetIndex'] == int(x['test_0'])], axis=1))

        array[item] = (array[item].apply(lambda x: np.multiply(np.resize(x,num_max_tracks),np.concatenate([np.ones(len(x)*(len(x) <= num_max_tracks) + num_max_tracks*(len(x) > num_max_tracks)),np.full((num_max_tracks-len(x))*(len(x) < num_max_tracks),np.nan, dtype='float32')]) )) ) 
        if item == "nn_track_pt":
            array_np = np.array([*array[item].to_numpy()])
            array_pt = np.array([*array[item].to_numpy()])
            track_sort_index = np.argsort(array_np)
        axis = 1
        index = list(np.ix_(*[np.arange(track_sort_index) for track_sort_index in array_np.shape]))
        array_np = np.array([*array[item].to_numpy()])
        index[axis] = (-array_pt).argsort(axis)
        x_data[0:array.shape[0],slice(counter_tracks+max_counter_cluster,max_counter_cluster+counter_tracks+((num_max_tracks-1)*num_track_variables)+1,num_track_variables)] = array_np[tuple(index)]

        counter_tracks = counter_tracks + 1
     
    counter_muons = 0
    max_counter_tracks =  max_counter_cluster+counter_tracks+((num_max_tracks-1)*num_track_variables)

    for item in (array.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values:
        array[item] = (array.apply(lambda x: x[item][x['nn_MSeg_jetIndex'] == int(x['test_0'])], axis=1))
        array[item] = (array[item].apply(lambda x: np.multiply(np.resize(x,num_max_muonSegs),np.concatenate([np.ones(len(x)*(len(x) <= num_max_muonSegs) + num_max_muonSegs*(len(x) > num_max_muonSegs)),np.full((num_max_muonSegs-len(x))*(len(x) < num_max_muonSegs),np.nan, dtype='float32')]) )) )
        array_np = np.array([*array[item].to_numpy()])
        x_data[0:array.shape[0],slice(counter_muons+max_counter_tracks,max_counter_tracks+counter_muons+((num_max_muonSegs-1)*num_muon_variables)+1,num_muon_variables)] = array_np
        counter_muons = counter_muons + 1


    np.set_printoptions(threshold=sys.maxsize)


    #print( (array.loc[:,'clus_pt':'clusTime']).columns.values + (array.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values + (array.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values ) 
    #print( (array.loc[:,'clus_pt':'clusTime']).columns.values)

    initial_names = ['label','mcEventWeight','flatWeight','jet_pt','jet_eta','jet_phi', 'jet_isClean_LooseBadLLP', 'aux_llp_Lxy','aux_llp_Lz','aux_llp_pt','aux_llp_eta','aux_llp_phi','llp_mH','llp_mS']

    constit_cols = ((array.loc[:,'clus_pt':'clusTime']).columns.values)
    constit_names = []
    for a in range(num_max_constits):
        for b in constit_cols:
            constit_names.append('%s_%d' % (b, a))

    track_cols = ((array.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values)
    track_names = []
    for a in range(num_max_tracks):
        for b in track_cols:
            track_names.append('%s_%d' % (b, a))

    MSeg_cols = ((array.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values)
    MSeg_names = []
    for a in range(num_max_muonSegs):
        for b in MSeg_cols:
            MSeg_names.append('%s_%d' % (b, a))

    total_names = initial_names + constit_names + track_names + MSeg_names

    final_dataFrame = pd.DataFrame(x_data, columns = total_names)
  
    return(final_dataFrame)

    #print(array)


def process_signal_events(array, llp_mH, llp_mS):
    #gives two arrays in 1, [0] corresponds to first LLP, [1] correspodns to second LLP
    array['test'] = array.apply(lambda x: ([((x['jet_eta'] - i)**2+(x['jet_phi'] - j)**2) for i,j in zip(x['LLP_eta'],x['LLP_phi'])]) , axis=1)
    #array['test_0'] = (array.test[array['test'].apply(lambda x: len(x[0])) > 0]).apply(lambda x: np.argmin((x[0])*(1+(-2)*(min(x[0]) > 0.04))))
    #test_0: LLP 0, test_1: LLP_1: only choose lowest DR, which is less than 0.4, make negative if higher
    array['test_0'] = (array.test[array['test'].apply(lambda x: len(x[0])) > 0]).apply(lambda x: np.argmin((x[0]))*(1+(-2)*(min(x[0])>0.4)*1))
    array['test_1'] = (array.test[array['test'].apply(lambda x: len(x[1])) > 0]).apply(lambda x: np.argmin((x[1]))*(1+(-2)*(min(x[1])>0.4)*1))
    #array['test_1'] = (array.test[array['test'].apply(lambda x: len(x[1])) > 0]).apply(lambda x: np.argmin((x[1])*(1+(-2)*(min(x[1]) > 0.04))))
    #TODO Look for ways to still see if two available jets when min is same jet
    #Choose only non-negative indices, and where it's not the same jet matched to both LLPs
    array['test_0'] = array.test_0[array.test_0 >=0]
    array['test_1'] = array.test_1[array.test_1 >=0]
    array['test_1'] = array['test_1'].loc[~(array['test_0'] == array['test_1'])]

    #print(array.test_0.loc[(array.test_0 >= 0) | (array.test_1 >= 0)] )
    #print(array.test_1.loc[(array.test_0 >= 0) | (array.test_1 >= 0)])

    #Select only rows with either LLPs are matched to a jet
    array = array.loc[ ((array.test_0 >= 0) | (array.test_1 >= 0))].copy()


    #print(array['LLP_eta'].apply(lambda x: ([((x[1] < 1.5 and i > 1400) or (x[1] > 1.5 and j > 3000)) for i,j in zip(x['LLP_eta'].apply( lambda x: x[1]),x['LLP_phi'].apply(lambda x: x[1]))])))
    #print( array.apply(lambda x: ([((x['LLP_eta'] < 1.5 and i > 1200) or (x['LLP_eta'] > 1.5 and j > 3000) for i,j in zip(x['LLP_Lxy'],x['LLP_Lz'])])) , axis=1))

    #Find rows with LLPs with valid eta/decay radius/z combos
    array['LLP_0_Lxy'] =  (array.LLP_Lxy.loc[array['LLP_eta'].apply(lambda x: abs(x[0])  ) < 1.4]).apply(lambda x: x[0] > 1500) 
    array['LLP_0_Lxy_end'] =  (array.LLP_Lxy.loc[array['LLP_eta'].apply(lambda x: abs(x[0])  ) < 1.4]).apply(lambda x: x[0] < 4000) 
    array['LLP_0_Lz'] =  (array.LLP_Lz.loc[array['LLP_eta'].apply(lambda x: abs(x[0])  ) > 1.4]).apply(lambda x: abs(x[0]) > 3000) 
    array['LLP_0_Lz_end'] =  (array.LLP_Lz.loc[array['LLP_eta'].apply(lambda x: abs(x[0])  ) > 1.4]).apply(lambda x: abs(x[0]) < 6000) 
    array['LLP_1_Lxy'] =  (array.LLP_Lxy.loc[array['LLP_eta'].apply(lambda x: abs(x[1])  ) < 1.4]).apply(lambda x: x[1] > 1500) 
    array['LLP_1_Lxy_end'] =  (array.LLP_Lxy.loc[array['LLP_eta'].apply(lambda x: abs(x[1])  ) < 1.4]).apply(lambda x: x[1] < 4000) 
    array['LLP_1_Lz'] =  (array.LLP_Lz.loc[array['LLP_eta'].apply(lambda x: abs(x[1])  ) > 1.4]).apply(lambda x: abs(x[1]) > 3000) 
    array['LLP_1_Lz_end'] =  (array.LLP_Lz.loc[array['LLP_eta'].apply(lambda x: abs(x[1])  ) > 1.4]).apply(lambda x: abs(x[1]) < 6000) 

    #print(array.test_0.loc[array.LLP_0_Lxy | array.LLP_0_Lz])
    #print(array.test_1.loc[array.LLP_1_Lxy | array.LLP_1_Lz])

    array_0 = array.loc[ ( (array.LLP_0_Lxy & array.LLP_0_Lxy_end) | (array.LLP_0_Lz & array.LLP_0_Lz_end)) & (~np.isnan(array.test_0))].copy()
    array_1 = array.loc[ ( (array.LLP_1_Lxy & array.LLP_1_Lxy_end) | (array.LLP_1_Lz & array.LLP_1_Lz_end) ) & (~np.isnan(array.test_1))].copy()

    #print(array_0)

    array_0['jet_pt_llp']  = array_0.apply(lambda x: ([(x['jet_pt'][(int(x['test_0']))]) ]) , axis=1)  
    array_0['jet_isClean_LooseBadLLP_llp']  = array_0.apply(lambda x: ([(x['jet_isClean_LooseBadLLP'][(int(x['test_0']))]) ]) , axis=1)  
    array_0['jet_eta_llp']  = (array_0.apply(lambda x: ([(x['jet_eta'][(int(x['test_0']))]) ]) , axis=1)  )
    array_0['jet_phi_llp']  = (array_0.apply(lambda x: ([(x['jet_phi'][(int(x['test_0']))]) ]) , axis=1)  )
    array_0['aux_llp_Lxy']  = (array_0.apply(lambda x: ([(x['LLP_Lxy'][0]) ]) , axis=1)  )
    array_0['aux_llp_Lz']  = (array_0.apply(lambda x: ([(x['LLP_Lz'][0]) ]) , axis=1)  )
    array_0['aux_llp_pt']  = (array_0.apply(lambda x: ([(x['LLP_pt'][0]) ]) , axis=1)  )
    array_0['aux_llp_eta']  = (array_0.apply(lambda x: ([(x['LLP_eta'][0]) ]) , axis=1)  )
    array_0['aux_llp_phi']  = (array_0.apply(lambda x: ([(x['LLP_phi'][0]) ]) , axis=1)  )

    array_1['jet_pt_llp']  = (array_1.apply(lambda x: ([(x['jet_pt'][(int(x['test_1']))]) ]) , axis=1)  )
    array_1['jet_isClean_LooseBadLLP_llp']  = (array_1.apply(lambda x: ([(x['jet_isClean_LooseBadLLP'][(int(x['test_1']))]) ]) , axis=1)  )
    array_1['jet_eta_llp']  = (array_1.apply(lambda x: ([(x['jet_eta'][(int(x['test_1']))]) ]) , axis=1)  )
    array_1['jet_phi_llp']  = (array_1.apply(lambda x: ([(x['jet_phi'][(int(x['test_1']))]) ]) , axis=1)  )
    array_1['aux_llp_Lxy']  = (array_1.apply(lambda x: ([(x['LLP_Lxy'][1]) ]) , axis=1)  )
    array_1['aux_llp_Lz']  = (array_1.apply(lambda x: ([(x['LLP_Lz'][1]) ]) , axis=1)  )
    array_1['aux_llp_pt']  = (array_1.apply(lambda x: ([(x['LLP_pt'][1]) ]) , axis=1)  )
    array_1['aux_llp_eta']  = (array_1.apply(lambda x: ([(x['LLP_eta'][1]) ]) , axis=1)  )
    array_1['aux_llp_phi']  = (array_1.apply(lambda x: ([(x['LLP_phi'][1]) ]) , axis=1)  )

    array_0['jet_pt_llp'] = array_0.jet_pt_llp.apply(lambda x: x[0]) 
    array_0['jet_isClean_LooseBadLLP_llp'] = array_0.jet_isClean_LooseBadLLP_llp.apply(lambda x: x[0]) 
    array_0['jet_eta_llp'] = array_0.jet_eta_llp.apply(lambda x: x[0]) 
    array_0['jet_phi_llp'] = array_0.jet_phi_llp.apply(lambda x: x[0]) 

    array_0['aux_llp_Lxy'] = array_0.aux_llp_Lxy.apply(lambda x: x[0]) 
    array_0['aux_llp_Lz'] = array_0.aux_llp_Lz.apply(lambda x: x[0]) 
    array_0['aux_llp_pt'] = array_0.aux_llp_pt.apply(lambda x: x[0]) 
    array_0['aux_llp_eta'] = array_0.aux_llp_eta.apply(lambda x: x[0]) 
    array_0['aux_llp_phi'] = array_0.aux_llp_phi.apply(lambda x: x[0]) 

    array_0 = array_0.loc[ array_0.jet_pt_llp >= 40000]
    array_0 = array_0.loc[ array_0.jet_pt_llp < 500000]
    array_0 = array_0.loc[ array_0.jet_eta_llp >= -2.5]
    array_0 = array_0.loc[ array_0.jet_eta_llp <= 2.5]

    array_1['jet_pt_llp'] = array_1.jet_pt_llp.apply(lambda x: x[0]) 
    array_1['jet_isClean_LooseBadLLP_llp'] = array_1.jet_isClean_LooseBadLLP_llp.apply(lambda x: x[0]) 
    array_1['jet_eta_llp'] = array_1.jet_eta_llp.apply(lambda x: x[0]) 
    array_1['jet_phi_llp'] = array_1.jet_phi_llp.apply(lambda x: x[0]) 

    array_1['aux_llp_Lxy'] = array_1.aux_llp_Lxy.apply(lambda x: x[0]) 
    array_1['aux_llp_Lz'] = array_1.aux_llp_Lz.apply(lambda x: x[0]) 
    array_1['aux_llp_pt'] = array_1.aux_llp_pt.apply(lambda x: x[0]) 
    array_1['aux_llp_eta'] = array_1.aux_llp_eta.apply(lambda x: x[0]) 
    array_1['aux_llp_phi'] = array_1.aux_llp_phi.apply(lambda x: x[0]) 

    array_1 = array_1.loc[ array_1.jet_pt_llp >= 40000]
    array_1 = array_1.loc[ array_1.jet_pt_llp < 500000]
    array_1 = array_1.loc[ array_1.jet_eta_llp >= -2.5]
    array_1 = array_1.loc[ array_1.jet_eta_llp <= 2.5]

    num_cluster_variables = len((array_0.loc[:,'clus_pt':'clusTime']).columns.values)
    num_track_variables = len((array_0.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values)
    num_muon_variables = len((array_0.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values)

    num_max_constits = 30
    num_max_tracks = 20
    num_max_muonSegs = 70

    size_0 = array_0.shape[0] + array_1.shape[0]
    size_1 = (num_cluster_variables*num_max_constits) + (num_track_variables*num_max_tracks) + (num_muon_variables*num_max_muonSegs) + 16
    x_data = np.full([size_0,size_1],np.nan, dtype='float32')
    

    x_data[0:array_0.shape[0],0] = np.ones(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],0] = np.ones(array_1.shape[0])

    x_data[0:array_0.shape[0],1] = np.ones(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],1] = np.ones(array_1.shape[0])

    x_data[0:array_0.shape[0],2] = np.ones(array_0.shape[0])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],2] = np.ones(array_1.shape[0])

    x_data[0:array_0.shape[0],3] = np.array([*array_0['jet_pt_llp'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],3] = np.array([*array_1['jet_pt_llp'].to_numpy()])

    x_data[0:array_0.shape[0],4] = np.array([*array_0['jet_eta_llp'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],4] = np.array([*array_1['jet_eta_llp'].to_numpy()])

    x_data[0:array_0.shape[0],5] = np.array([*array_0['jet_phi_llp'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],5] = np.array([*array_1['jet_phi_llp'].to_numpy()])

    x_data[0:array_0.shape[0],6] = np.array([*array_0['jet_isClean_LooseBadLLP_llp'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],6] = np.array([*array_1['jet_isClean_LooseBadLLP_llp'].to_numpy()])

    x_data[0:array_0.shape[0],7] = np.array([*array_0['aux_llp_Lxy'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],7] = np.array([*array_1['aux_llp_Lxy'].to_numpy()])
  
    x_data[0:array_0.shape[0],8] = np.array([*array_0['aux_llp_Lz'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],8] = np.array([*array_1['aux_llp_Lz'].to_numpy()])

    x_data[0:array_0.shape[0],9] = np.array([*array_0['aux_llp_pt'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],9] = np.array([*array_1['aux_llp_pt'].to_numpy()])

    x_data[0:array_0.shape[0],10] = np.array([*array_0['aux_llp_eta'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],10] = np.array([*array_1['aux_llp_eta'].to_numpy()])

    x_data[0:array_0.shape[0],11] = np.array([*array_0['aux_llp_phi'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],11] = np.array([*array_1['aux_llp_phi'].to_numpy()])

    x_data[0:array_0.shape[0],12] = np.ones(array_0.shape[0])*llp_mH
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],12] = np.ones(array_1.shape[0])*llp_mH

    x_data[0:array_0.shape[0],13] = np.ones(array_0.shape[0])*llp_mS
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],13] = np.ones(array_1.shape[0])*llp_mS

    x_data[0:array_0.shape[0],14] = np.array([*array_0['eventNumber'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],14] = np.array([*array_1['eventNumber'].to_numpy()])

    x_data[0:array_0.shape[0],15] = np.array([*array_0['runNumber'].to_numpy()])
    x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],15] = np.array([*array_1['runNumber'].to_numpy()])

    clus_sort_index_0 = np.zeros(num_max_constits)
    track_sort_index_0 = np.zeros(num_max_constits)
    clus_sort_index_1 = np.zeros(num_max_tracks)
    track_sort_index_1 = np.zeros(num_max_tracks)
    #print(array_0.clus_pt.apply(lambda x: len(x)))
    counter_cluster=16
    for item in (array_0.loc[:,'clus_pt':'clusTime']).columns.values:
        array_0[item] = (array_0.apply(lambda x: x[item][x['cluster_jetIndex'] == int(x['test_0'])], axis=1))
        #SORRY ABOUT THIS IT IS BAD CODE
        array_0[item] = (array_0[item].apply(lambda x: np.multiply(np.resize(x,num_max_constits),np.concatenate([np.ones(len(x)*(len(x) <= num_max_constits) + num_max_constits*(len(x) > num_max_constits)),np.full((num_max_constits-len(x))*(len(x) < num_max_constits),np.nan, dtype='float32')]) )) ) 
        if item == "clus_pt":
            array_0_np = np.array([*array_0[item].to_numpy()])
            array_0_pt = np.array([*array_0[item].to_numpy()])
            clus_sort_index_0 = np.argsort(array_0_np)
        axis = 1
        index = list(np.ix_(*[np.arange(clus_sort_index_0) for clus_sort_index_0 in array_0_np.shape]))
        array_0_np = np.array([*array_0[item].to_numpy()])
        index[axis] = (-array_0_pt).argsort(axis)
        x_data[0:array_0.shape[0],slice(counter_cluster,counter_cluster+((num_max_constits-1)*num_cluster_variables)+1,num_cluster_variables)] = array_0_np[tuple(index)]
        #print(array_0[item].to_numpy().shape)
        #print(x_data[:,slice(0,112,28)].shape)
        #print((array_0[item].to_numpy()).shape)
        #test[0:array_0.shape[0],0:20] = array_0[item].to_numpy()
        #test = np.asarray(array_0[item].to_numpy()).copy()


        #print( np.pad(array_0[item].to_numpy(),(20),mode='constant', constant_values=0).shape )
        #x_data[:,slice(0,28*20,28)] = array_0[item].to_numpy()
        #print(array_0[item])
        #array_1[item] = (array_1.apply(lambda x: x[item][x['cluster_jetIndex'] == int(x['test_1'])], axis=1))
        array_1[item] = (array_1.apply(lambda x: x[item][x['cluster_jetIndex'] == int(x['test_1'])], axis=1))
        #SORRY ABOUT THIS IT IS BAD CODE
        array_1[item] = (array_1[item].apply(lambda x: np.multiply(np.resize(x,num_max_constits),np.concatenate([np.ones(len(x)*(len(x) <= num_max_constits) + num_max_constits*(len(x) > num_max_constits)),np.full((num_max_constits-len(x))*(len(x) < num_max_constits),np.nan, dtype='float32')]) )) ) 
        if item == "clus_pt":
            array_1_np = np.array([*array_1[item].to_numpy()])
            array_1_pt = np.array([*array_1[item].to_numpy()])
            clus_sort_index_1 = np.argsort(array_1_np)
        axis = 1
        index = list(np.ix_(*[np.arange(clus_sort_index_1) for clus_sort_index_1 in array_1_np.shape]))
        array_1_np = np.array([*array_1[item].to_numpy()])
        index[axis] = (-array_1_pt).argsort(axis)
        x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],slice(counter_cluster,counter_cluster+((num_max_constits-1)*num_cluster_variables)+1,num_cluster_variables)] = array_1_np[tuple(index)]
        counter_cluster = counter_cluster+1

    counter_tracks=0
    max_counter_cluster = counter_cluster+((num_max_constits-1)*num_cluster_variables)

    for item in (array_0.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values:
        array_0[item] = (array_0.apply(lambda x: x[item][x['nn_track_jetIndex'] == int(x['test_0'])], axis=1))

        array_0[item] = (array_0[item].apply(lambda x: np.multiply(np.resize(x,num_max_tracks),np.concatenate([np.ones(len(x)*(len(x) <= num_max_tracks) + num_max_tracks*(len(x) > num_max_tracks)),np.full((num_max_tracks-len(x))*(len(x) < num_max_tracks),np.nan, dtype='float32')]) )) ) 
        if item == "nn_track_pt":
            array_0_np = np.array([*array_0[item].to_numpy()])
            array_0_pt = np.array([*array_0[item].to_numpy()])
            track_sort_index_0 = np.argsort(array_0_np)
        axis = 1
        index = list(np.ix_(*[np.arange(track_sort_index_0) for track_sort_index_0 in array_0_np.shape]))
        array_0_np = np.array([*array_0[item].to_numpy()])
        index[axis] = (-array_0_pt).argsort(axis)
        x_data[0:array_0.shape[0],slice(counter_tracks+max_counter_cluster,max_counter_cluster+counter_tracks+((num_max_tracks-1)*num_track_variables)+1,num_track_variables)]  = array_0_np[tuple(index)]

        array_1[item] = (array_1.apply(lambda x: x[item][x['nn_track_jetIndex'] == int(x['test_1'])], axis=1))

        array_1[item] = (array_1[item].apply(lambda x: np.multiply(np.resize(x,num_max_tracks),np.concatenate([np.ones(len(x)*(len(x) <= num_max_tracks) + num_max_tracks*(len(x) > num_max_tracks)),np.full((num_max_tracks-len(x))*(len(x) < num_max_tracks),np.nan, dtype='float32')]) )) ) 
        if item == "nn_track_pt":
            array_1_np = np.array([*array_1[item].to_numpy()])
            array_1_pt = np.array([*array_1[item].to_numpy()])
            track_sort_index_1 = np.argsort(array_1_np)
        axis = 1
        index = list(np.ix_(*[np.arange(track_sort_index_1) for track_sort_index_1 in array_1_np.shape]))
        array_1_np = np.array([*array_1[item].to_numpy()])
        index[axis] = (-array_1_pt).argsort(axis)
        x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],slice(counter_tracks+max_counter_cluster,max_counter_cluster+counter_tracks+((num_max_tracks-1)*num_track_variables)+1,num_track_variables)]  = array_1_np[tuple(index)]

        counter_tracks = counter_tracks + 1
     
    counter_muons = 0
    max_counter_tracks =  max_counter_cluster+counter_tracks+((num_max_tracks-1)*num_track_variables)

    for item in (array_0.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values:
        array_0[item] = (array_0.apply(lambda x: x[item][x['nn_MSeg_jetIndex'] == int(x['test_0'])], axis=1))
        array_0[item] = (array_0[item].apply(lambda x: np.multiply(np.resize(x,num_max_muonSegs),np.concatenate([np.ones(len(x)*(len(x) <= num_max_muonSegs) + num_max_muonSegs*(len(x) > num_max_muonSegs)),np.full((num_max_muonSegs-len(x))*(len(x) < num_max_muonSegs),np.nan, dtype='float32')]) )) )
        array_0_np = np.array([*array_0[item].to_numpy()])
        x_data[0:array_0.shape[0],slice(counter_muons+max_counter_tracks,max_counter_tracks+counter_muons+((num_max_muonSegs-1)*num_muon_variables)+1,num_muon_variables)] = array_0_np

        array_1[item] = (array_1.apply(lambda x: x[item][x['nn_MSeg_jetIndex'] == int(x['test_1'])], axis=1))

        array_1[item] = (array_1[item].apply(lambda x: np.multiply(np.resize(x,num_max_muonSegs),np.concatenate([np.ones(len(x)*(len(x) <= num_max_muonSegs) + num_max_muonSegs*(len(x) > num_max_muonSegs)),np.full((num_max_muonSegs-len(x))*(len(x) < num_max_muonSegs),np.nan, dtype='float32')]) )) ) 
        array_1_np = np.array([*array_1[item].to_numpy()])
        x_data[array_0.shape[0]:array_0.shape[0]+array_1.shape[0],slice(counter_muons+max_counter_tracks,counter_muons+max_counter_tracks+((num_max_muonSegs-1)*num_muon_variables)+1,num_muon_variables)] = array_1_np

        counter_muons = counter_muons + 1




    #print( (array_0.loc[:,'clus_pt':'clusTime']).columns.values + (array_0.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values + (array_0.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values ) 
    #print( (array_0.loc[:,'clus_pt':'clusTime']).columns.values)

    initial_names = ['label','mcEventWeight','flatWeight','jet_pt','jet_eta','jet_phi','jet_isClean_LooseBadLLP','aux_llp_Lxy','aux_llp_Lz','aux_llp_pt','aux_llp_eta','aux_llp_phi','llp_mH','llp_mS', 'eventNumber', 'runNumber']

    constit_cols = ((array_0.loc[:,'clus_pt':'clusTime']).columns.values)
    constit_names = []
    for a in range(num_max_constits):
        for b in constit_cols:
            constit_names.append('%s_%d' % (b, a))

    track_cols = ((array_0.loc[:,'nn_track_pt':'nn_track_SCTHits']).columns.values)
    track_names = []
    for a in range(num_max_tracks):
        for b in track_cols:
            track_names.append('%s_%d' % (b, a))

    MSeg_cols = ((array_0.loc[:,'nn_MSeg_etaPos':'nn_MSeg_t0']).columns.values)
    MSeg_names = []
    for a in range(num_max_muonSegs):
        for b in MSeg_cols:
            MSeg_names.append('%s_%d' % (b, a))

    total_names = initial_names + constit_names + track_names + MSeg_names

    final_dataFrame = pd.DataFrame(x_data, columns = total_names)
  
    return(final_dataFrame)

    #print(x_data[42])
    #array_0['clus_pt'] = array_0.clus_pt[array_0.clus_pt > 0]

    #print( array.LLP_1_Lz )
    #print(array['test_1'])
    #print(array.LLP_eta[array['test'].apply(lambda x: len(x[0])) > 0]) 

    #print(array.LLP_eta[array.test_1 >= 0 and ~(array['test_0'] == array['test_1'])])
    #print(array.loc[array['test'] == True])
    '''
    print(array.jet_eta.iloc[500])
    print(array.LLP_eta.iloc[500])
    print(array.jet_phi.iloc[500])
    print(array.LLP_phi.iloc[500])
    print(array['test'].iloc[500][0])
    '''

if __name__ == '__main__':

    name, filename, data_type, date_time = sys.argv

    executor = concurrent.futures.ThreadPoolExecutor(64)
    print(multiprocessing.cpu_count())

    print("Loading data")
    start = time.time()

    var_list_QCD = ["HLT_jet_TAU60","HLT_jet_TAU100","HLT_jet_LLPNM","HLT_jet_LLPRO","HLT_jet_isBIB","HLT_jet_eta","HLT_jet_phi","mcEventWeight","signal","QCD","BIB","jzN","jet_pt", "jet_eta", "jet_phi", "jet_isClean_LooseBadLLP", "jet_E", "jet_index","cluster_jetIndex","clus_pt","clus_eta","clus_phi","e_PreSamplerB","e_EMB1","e_EMB2","e_EMB3","e_PreSamplerE","e_EME1","e_EME2","e_EME3","e_HEC0","e_HEC1","e_HEC2","e_HEC3","e_TileBar0","e_TileBar1","e_TileBar2","e_TileGap1","e_TileGap2","e_TileGap3","e_TileExt0","e_TileExt1","e_TileExt2","e_FCAL0","e_FCAL1","e_FCAL2","clusTime","nn_track_jetIndex","nn_track_pt","nn_track_eta","nn_track_phi","nn_track_d0","nn_track_z0","nn_track_PixelShared","nn_track_PixelSplit","nn_track_SCTShared","nn_track_PixelHoles","nn_track_SCTHoles","nn_track_PixelHits","nn_track_SCTHits","nn_MSeg_etaPos","nn_MSeg_phiPos","nn_MSeg_etaDir","nn_MSeg_phiDir","nn_MSeg_t0","nn_MSeg_jetIndex"]

    var_list_MC = ["HLT_jet_TAU60","HLT_jet_TAU100","HLT_jet_LLPNM","HLT_jet_LLPRO","HLT_jet_isBIB","HLT_jet_eta","HLT_jet_phi","mcEventWeight","LLP_pt","LLP_eta","LLP_phi","LLP_Lxy","LLP_Lz","signal","QCD","BIB","jzN","jet_pt", "jet_eta", "jet_phi", "jet_isClean_LooseBadLLP", "jet_E", "jet_index","cluster_jetIndex","clus_pt","clus_eta","clus_phi","e_PreSamplerB","e_EMB1","e_EMB2","e_EMB3","e_PreSamplerE","e_EME1","e_EME2","e_EME3","e_HEC0","e_HEC1","e_HEC2","e_HEC3","e_TileBar0","e_TileBar1","e_TileBar2","e_TileGap1","e_TileGap2","e_TileGap3","e_TileExt0","e_TileExt1","e_TileExt2","e_FCAL0","e_FCAL1","e_FCAL2","clusTime","nn_track_jetIndex","nn_track_pt","nn_track_eta","nn_track_phi","nn_track_d0","nn_track_z0","nn_track_PixelShared","nn_track_PixelSplit","nn_track_SCTShared","nn_track_PixelHoles","nn_track_SCTHoles","nn_track_PixelHits","nn_track_SCTHits","nn_MSeg_etaPos","nn_MSeg_phiPos","nn_MSeg_etaDir","nn_MSeg_phiDir","nn_MSeg_t0","nn_MSeg_jetIndex", "eventNumber", "runNumber"]

    var_list_data = ["HLT_jet_TAU60","HLT_jet_TAU100","HLT_jet_LLPNM","HLT_jet_LLPRO","HLT_jet_isBIB","HLT_jet_eta","HLT_jet_phi","HLT_jet_eta","HLT_jet_phi","signal","QCD","BIB","jzN","jet_pt", "jet_eta", "jet_phi", "jet_isClean_LooseBadLLP", "jet_E", "jet_index","cluster_jetIndex","clus_pt","clus_eta","clus_phi","e_PreSamplerB","e_EMB1","e_EMB2","e_EMB3","e_PreSamplerE","e_EME1","e_EME2","e_EME3","e_HEC0","e_HEC1","e_HEC2","e_HEC3","e_TileBar0","e_TileBar1","e_TileBar2","e_TileGap1","e_TileGap2","e_TileGap3","e_TileExt0","e_TileExt1","e_TileExt2","e_FCAL0","e_FCAL1","e_FCAL2","clusTime","nn_track_jetIndex","nn_track_pt","nn_track_eta","nn_track_phi","nn_track_d0","nn_track_z0","nn_track_PixelShared","nn_track_PixelSplit","nn_track_SCTShared","nn_track_PixelHoles","nn_track_SCTHoles","nn_track_PixelHits","nn_track_SCTHits","nn_MSeg_etaPos","nn_MSeg_phiPos","nn_MSeg_etaDir","nn_MSeg_phiDir","nn_MSeg_t0","nn_MSeg_jetIndex"]

    df = pd.DataFrame()
    print("DATA TYPE: " + str(int(data_type)) )

    if (int(data_type) == 1):

        for path,entries_start,entries_stop,arrays in uproot.iterate(filename, "trees_msVtx_",
            var_list_MC,entrysteps=50000,executor=executor, outputtype=pd.DataFrame, reportpath=True, reportentries=True):
            llp_mH = int(find_number(path,"mH")[0])
            llp_mS = int(find_number(path,"mS")[0])
            signal_arrays = arrays[arrays.signal == 1]
            QCD_arrays = arrays[arrays.QCD == 1]
            BIB_arrays = arrays[arrays.BIB == 1]

            if len(signal_arrays) > 0:
                df_signal = process_signal_events(signal_arrays, llp_mH, llp_mS)
                df = df.append(df_signal, ignore_index=False)
       
        df.to_pickle("outputs/signal_"+ str(date_time) + "_raw_output.pkl")

    if (int(data_type) == 0):

        for path,entries_start,entries_stop,arrays in uproot.iterate(filename, "trees_msVtx_",
            var_list_QCD,entrysteps=50000,executor=executor, outputtype=pd.DataFrame, reportpath=True, reportentries=True):
            signal_arrays = arrays[arrays.signal == 1]
            QCD_arrays = arrays[arrays.QCD == 1]
            BIB_arrays = arrays[arrays.BIB == 1]
            JZ_num = int(find_number(path,"JZ")[0])

            if len(QCD_arrays) > 0:
                df_signal = process_qcd_events(QCD_arrays)
                df = df.append(df_signal, ignore_index=False)

        df.to_pickle("outputs/qcd_"+ str(JZ_num) + "_" + str(date_time) + "_raw_output.pkl")

    if (int(data_type) == 2):

        for path,entries_start,entries_stop,arrays in uproot.iterate(filename, "trees_msVtx_",
            var_list_data,entrysteps=50000,executor=executor, outputtype=pd.DataFrame, reportpath=True, reportentries=True):
            signal_arrays = arrays[arrays.signal == 1]
            QCD_arrays = arrays[arrays.QCD == 1]
            BIB_arrays = arrays[arrays.BIB == 1]

            if len(BIB_arrays) > 0:
                df_signal = process_bib_events(BIB_arrays)
                df = df.append(df_signal, ignore_index=False)

        df.to_pickle("outputs/bib_"+ str(date_time) + "_raw_output.pkl")



