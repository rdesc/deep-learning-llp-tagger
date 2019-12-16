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

import tensorflow as tf
from keras.backend import tensorflow_backend as K

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Highway, Dropout, Masking, CuDNNLSTM, Convolution1D, Convolution2D, Flatten, Input, Embedding, LSTM
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.regularizers import l1, l2, L1L2
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model

import itertools

import sys

import glob

import time
import re

import argparse
import subprocess

from datetime import datetime

import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

def find_number(text, c):
    return re.findall(r'%s(\d+)' % c, text)

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


def analyse_roc_distanceTest(roc_files_distanceTest, filename):

    frac = 1.0
    num_max_constits=30
    num_max_tracks=20
    num_max_MSegs=30
    num_constit_lstm=60
    num_track_lstm=60
    num_mseg_lstm=25
    reg_value=0.001
    dropout_value = 0.1
    epochs = 50
    model_to_do = "lstm_test"
    doTrackLSTM = True
    doMSegLSTM = True
    doParametrization = False
    learning_rate = 0.002

    df = pd.read_pickle(filename)
    #Replace all NaN by 0
    df = df.fillna(0)



    #Delete some 'virtual' variables only needed for pre-processing
    del df['track_sign']
    del df['clus_sign']

    #I labelled all truth variables (except parametrization) with aux_*, so remove all those!!!
    '''
    auxDelete = [col for col in df if col.startswith("aux")]
    for item in auxDelete:
        del df[item]
    '''

    #TODO: not leave hardcoded
    #Decides if time is a variable or not
    deleteTime = False

    #Delete time variable for clusters and muon segments
    if deleteTime:
        clus_timeDelete = [col for col in df if col.startswith("clus_time")]
        for item in clus_timeDelete:
            del df[item]

        segTimeDelete = [col for col in df  if col.startswith("nn_MSeg_t0")]
        for item in segTimeDelete:
            del df[item]

    vertex_delete_x = [col for col in df if col.startswith("nn_track_vertex_x")]
    for item in vertex_delete_x:
        del df[item]
    vertex_delete_y = [col for col in df if col.startswith("nn_track_vertex_y")]
    for item in vertex_delete_y:
        del df[item]
    vertex_delete_z = [col for col in df if col.startswith("nn_track_vertex_z")]
    for item in vertex_delete_z:
        del df[item]

    print("Length of Signal is: " + str(df[df.label==1].shape[0]) )
    print("Length of QCD is: " + str(df[df.label==0].shape[0]) )
    print("Length of BIB is: " + str(df[df.label==2].shape[0]) )

    #Extract true label from input dataFrame
    Y = df['label']
    #Use pt flattened weights from pre-processing for weights
    weights = df['flatWeight']
    #Keep mcWeights for evaluation
    mcWeights = df['mcEventWeight']
    #Hard code start and end of names of variables
    X= df.loc[:,'jet_pt':'nn_MSeg_t0_29']
    #TODO: have true/false input to decide if we are deleting input parametrized variables
    del X['llp_mS']
    del X['llp_mH']
    #Delete variables we don't need
    del X['jet_isClean_LooseBadLLP']
    #Label Z as parametrized variables
    Z = df.loc[:,'llp_mH':'llp_mS']


    #Divide into train/test datasets
    X_train, X_test, y_train, y_test, weights_train, weights_test, mcWeights_train, mcWeights_test,  Z_train, Z_test = train_test_split(X, Y, weights, mcWeights, Z, test_size = 0.2)

    #Only keep the fraction of events to train specified as input
    X_train = X_train.iloc[0:int(X_train.shape[0]*frac)]
    y_train = y_train.iloc[0:int(y_train.shape[0]*frac)]
    weights_train = weights_train.iloc[0:int(weights_train.shape[0]*frac)]
    mcWeights_train = mcWeights_train.iloc[0:int(mcWeights_train.shape[0]*frac)]
    Z_train = Z_train.iloc[0:int(Z_train.shape[0]*frac)]

    #Divide testing set into epoch-by-epoch validation and final evaluation sets
    X_test, X_val, y_test, y_val, weights_test, weights_val, mcWeights_test, mcWeights_val, Z_test, Z_val = train_test_split(X_test, y_test, weights_test, mcWeights_test,  Z_test, test_size = 0.5)

    #Delete variables we don't need anymore (need to save memory...)
    del X
    del Y
    del Z
    del X_train
    del y_train
    del Z_train
    del X_test
    del y_test
    del Z_test
    
    #Hard-coded the number of variables inside each jet constituent, track or muon Segment
    #It is reduced by 1 for constituents and muon segments if time is not included
    num_constit_vars = 12
    if deleteTime == True:
        num_constit_vars = 11
    num_track_vars = 13
    num_MSeg_vars = 6
    if deleteTime == True:
        num_MSeg_vars = 5
    
    num_jet_vars = 3
    
            #Does same constructing for validation dataset
    if deleteTime:
        X_val_constit = X_val.loc[:,'clus_pt_0':'clus_l4hcal_'+str(num_max_constits-1)]
    else:
        X_val_constit = X_val.loc[:,'clus_pt_0':'clus_time_'+str(num_max_constits-1)]
    X_val_track = X_val.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
    if deleteTime:
        X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_chiSquared_'+str(num_max_MSegs-1)]
    else:
        X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
    X_val_jet = X_val.loc[:,'jet_pt':'jet_phi']
    if doParametrization:
        X_val_jet= X_val_jet.join(Z_val)
    
    X_val_constit = X_val_constit.values.reshape(X_val_constit.shape[0],num_max_constits,num_constit_vars)
    X_val_track = X_val_track.values.reshape(X_val_track.shape[0],num_max_tracks,num_track_vars)
    X_val_MSeg = X_val_MSeg.values.reshape(X_val_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)
    #Need to tell network shape of input for jet constituents
    constit_input = Input(shape=(X_val_constit[0].shape),dtype='float32',name='constit_input')
    #Have one LSTM layer, with regularizer, tied to input node
    constit_out = CuDNNLSTM(num_constit_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(constit_input)
    #Have a constit LSTM output, which does the classification using only constituents.
    #This lets you monitor how the consituent LSTM is doing, but is not used for anything else
    constit_output = Dense(3, activation='softmax', name='constit_output')(constit_out)
    
    #Need to tell network shape of input for tracks
    track_input = Input(shape=(X_val_track[0].shape),dtype='float32',name='track_input')
    #Have one LSTM layer, with regularizer, tied to input node
    track_out = CuDNNLSTM(num_track_lstm , kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(track_input)
    #Have a track LSTM output, which does the classification using only tracks.
    #This lets you monitor how the track LSTM is doing, but is not used for anything else
    track_output = Dense(3, activation='softmax', name='track_output')(track_out)
    
    #Need to tell network shape of input for muon segments
    MSeg_input = Input(shape=(X_val_MSeg[0].shape),dtype='float32',name='MSeg_input')
    #Have one LSTM layer, with regularizer, tied to input node
    MSeg_out = CuDNNLSTM(num_mseg_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(MSeg_input)
    #Have a muon segment LSTM output, which does the classification using only muon segments.
    #This lets you monitor how the muon segment LSTM is doing, but is not used for anything else
    MSeg_output = Dense(3, activation='softmax', name='MSeg_output')(MSeg_out)
    
    #Have an input for jet variables (pt, eta, phi)
    #Can add parametrization at this point
    jet_input = Input(shape = X_val_jet.values[0].shape, name='jet_input')
    #This is just a dense layer, not LSTM
    jet_out = Dense(3)(jet_input)
    jet_output = Dense(3, activation='softmax', name='jet_output')(jet_out)
    
    #Concatenate the LSTM nodes of constituents, tracks, muon segments and jet dense layer
    layersToConcatenate = [constit_out, track_out, MSeg_out, jet_input]
    
    #Only concatenate layers if we are actually using them
    if (doTrackLSTM and not doMSegLSTM):
        layersToConcatenate = [constit_out, track_out, jet_input]
    if (doMSegLSTM and not doTrackLSTM):
        layersToConcatenate = [constit_out, MSeg_out, jet_input]
    if (not doTrackLSTM and not doMSegLSTM):
        layersToConcatenate = [constit_out, jet_input]
    
    #Actually do the concatenation
    x = keras.layers.concatenate(layersToConcatenate)
    
    #Add a few dense layers after LSTM to connect the results of that together
    #TODO: optimise, understand this
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_value)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_value)(x)
    
    #Main output for 3 classes, must be softmax for multiclass
    main_output = Dense(3, activation='softmax', name='main_output')(x)
    
    layers_to_input = [constit_input, track_input, MSeg_input, jet_input]
    layers_to_output = [main_output, constit_output, track_output, MSeg_output, jet_output]
    weights_for_loss = [1., 0.1, 0.4, 0.2,0.1]
    x_to_validate = [X_val_constit, X_val_track, X_val_MSeg, X_val_jet.values]
    
    if (doTrackLSTM and not doMSegLSTM):
        layers_to_input = [constit_input, track_input,  jet_input]
        layers_to_output = [main_output, constit_output, track_output, jet_output]
        x_to_validate = [X_val_constit, X_val_track, X_val_jet.values]
        weights_for_loss = [1., 0.1, 0.4, 0.1]
    if (doMSegLSTM and not doTrackLSTM):
        print("HERE")
        layers_to_input = [constit_input, MSeg_input,  jet_input]
        layers_to_output = [main_output, constit_output, MSeg_output, jet_output]
        x_to_validate = [X_val_constit, X_val_MSeg, X_val_jet.values]
        weights_for_loss = [1., 0.1, 0.2,0.1]
    if (not doTrackLSTM and not doMSegLSTM):
        layers_to_input = [constit_input,  jet_input]
        layers_to_output = [main_output, constit_output, jet_output]
        x_to_validate = [X_val_constit, X_val_jet.values]
        weights_for_loss = [1., 0.1, 0.1]
    
    model = Model(inputs=layers_to_input, outputs=layers_to_output)
    signal = X_val[y_val == 1].dropna()
    
    
    opt = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                loss_weights=weights_for_loss, metrics=['accuracy'])

    colors = ['red','blue','green','pink','yellow']
    counter = 0

    for training_file in roc_files_distanceTest: 
        #Up to here same procedure as in training
        #Now load weights from already trained network
        model.load_weights("keras_outputs/"+training_file+'/checkpoint')

        #Use network to predict labels from test set, depending on if dense or lstm model
        prediction = model.predict(x_to_validate, verbose = True)
        prediction = prediction[0]

        signal_right_1 = prediction[:,1] > prediction[:,0]# and prediction[:,1] > prediction[:,2]
        qcd_right_1 = prediction[:,0] > prediction[:,1]# and prediction[:,0] > prediction[:,2]
        bib_right_1 = prediction[:,2] > prediction[:,0]# and prediction[:,2] > prediction[:,1]
    
        signal_right_2 = prediction[:,1] > prediction[:,2]
        qcd_right_2 = prediction[:,0] > prediction[:,2]
        bib_right_2 = prediction[:,2] > prediction[:,1]
    
    
        signal_highest = signal_right_1 & signal_right_2
        qcd_highest = qcd_right_1 & qcd_right_2
        bib_highest = bib_right_1 & bib_right_2
    
        signal_right = (1*(signal_highest+y_val == 2))[y_val == 1]
        qcd_right = (1*(qcd_highest+y_val == 1))[y_val == 0]
        bib_right = (1*(bib_highest+y_val == 3))[y_val == 2]

        print(signal_right.shape)
        print(signal.values.flatten().shape)

        signal_filter = signal['aux_llp_Lxy']
        bins = range(550,4000,50)
        llp_Lxy = int(find_number(training_file,"Lxy")[0])
        llp_Lz = int(find_number(training_file,"Lz")[0])
        g = sns.regplot(x=signal_filter.values.flatten(), y=1-signal_right, x_bins=bins, fit_reg=None, color=colors[counter], label=f"Lxy: {(llp_Lxy):.0f}" + f", Lz: {(llp_Lz):.0f}")
        g.set(yscale="log")
        counter = counter + 1
        plt.ylabel("1-Accuracy")
        plt.xlabel("Lxy")

    plt.legend()
    plt.savefig("plots/Lxy_studies/comparison.pdf", format = 'pdf', transparent = True)













