import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')

import tensorflow as tf
from keras.backend import tensorflow_backend as K

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Highway, Dropout, Masking, CuDNNLSTM, Convolution1D, Convolution2D, Flatten, Input, Embedding, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.regularizers import l1, l2, L1L2
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import sklearn
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from random import shuffle, seed


from random import shuffle
import pdb
import sklearn
from sklearn.preprocessing import minmax_scale
import math
import matplotlib.pyplot as plt

from evaluate_training import *

from datetime import datetime

os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'
os.environ['exception_verbosity']='high'

'''
Main function for DeepJet algorithm

'''
def train_llp_cnn(filename, useGPU2, frac = 1.0, num_max_constits=30, num_max_tracks=20, num_max_MSegs=30, num_constit_cnn=[64,32,32,8], num_track_cnn=[64,32,32,8], num_mseg_cnn=[32,16,4], model_to_do = "cnn_test"):
    
    # TODO: Delete time?
    # TODO: with parametrization?
    # TODO: make into args?
    num_constit_vars = 12
    num_track_vars = 13
    num_MSeg_vars = 6
    num_jet_vars = 3

    # Choose GPU
    if (useGPU2):
        os.environ["CUDA_VISIBLE_DEVICES"]="1"

    # TODO: Move to utils?
    # Create directories
    dirName = "plots/" + model_to_do
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        os.mkdir("plots/trainingDiagrams/" + model_to_do)
        os.mkdir("keras_outputs/" + model_to_do)
        print("Directory " , dirName ,  " Created")
    else:
        print("Directory " , dirName ,  " already exists")

    destination = "plots/" + model_to_do + "/"

    # Write a file with some details of architecture, will append final stats at end of training
    f = open(destination + "training_details.txt", "w+")
    f.write("\nnum_max_constits = %s\nnum_max_tracks = %s\nnum_max_MSegs = %s\nnum_constit_cnn = %s\nnum_track_cnn = %s\nnum_mseg_cnn = %s\n" % (num_max_constits, num_max_tracks, num_max_MSegs, num_constit_cnn, num_track_cnn, num_mseg_cnn))
    f.close()

    # Print these stats to stdout
    print("\nnum_max_constits = %s\nnum_max_tracks = %s\nnum_max_MSegs = %s\nnum_constit_cnn = %s\nnum_track_cnn = %s\nnum_mseg_cnn = %s\n" % (num_max_constits, num_max_tracks, num_max_MSegs, num_constit_cnn, num_track_cnn, num_mseg_cnn))

    #Keras magic
    # TODO: dryrun option?
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=64)
    tf.set_random_seed(1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)

    # Get input data
    df = pd.read_pickle(filename)
    # Replace infs with and nans with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Delete some 'virtual' variables only needed for pre-processing
    del df['track_sign']
    del df['clus_sign']

    # Delete track_vertex vars in tracks
    vertex_delete = [col for col in df if col.startswith("nn_track_vertex_x")]
    vertex_delete += [col for col in df if col.startswith("nn_track_vertex_y")]
    vertex_delete += [col for col in df if col.startswith("nn_track_vertex_z")]
    for item in vertex_delete:
        del df[item]

    # Print sizes of inputs for signal, qcd, and bib
    print("\nLength of Signal is: " + str(df[df.label==1].shape[0]))
    print("Length of QCD is: " + str(df[df.label==0].shape[0]))
    print("Length of BIB is: " + str(df[df.label==2].shape[0]))

    # Extract labels
    Y = df['label']
    # Use pt flattened weights from pre-processing for weights
    weights = df['flatWeight'] # TODO: what are these weights for?
    # Keep mcWeights TODO: what is this? for evaluation
    mcWeights = df['mcEventWeight']
    # Hard code start and end of names of variables # TODO: test in iPython
    X = df.loc[:,'clus_pt_0':'nn_MSeg_t0_29']
    X = df.loc[:,'jet_pt':'jet_phi'].join(X)

    # Label Z as parametrized variables
    Z = df.loc[:,'llp_mH':'llp_mS']
    mass_array = (df.groupby(['llp_mH','llp_mS']).size().reset_index().rename(columns={0:'count'}))
    mass_array['proportion'] = mass_array['count']/len(df.index)

    # TODO: add many of these steps to pre_processing
    # Save memory
    del df

    # Split data into train/test datasets
    X_train, X_test, y_train, y_test, weights_train, weights_test, mcWeights_train, mcWeights_test, Z_train, Z_test = train_test_split(X, Y, weights, mcWeights, Z, test_size = 0.2)

    # Keep fraction of events specified by frac param
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

    #Convert labels to categorical (needed for multiclass training)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    # TODO: add function for each setup tracks, mseg, constit
    # TODO: add function to avoid repeat code for splitting up train, test, and val
    # Split X into track, MSeg, and constit inputs
    X_train_constit = X_train.loc[:,'clus_pt_0':'clus_time_'+str(num_max_constits-1)]
    X_train_track = X_train.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
    X_train_MSeg = X_train.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
    X_train_jet = X_train.loc[:,'jet_pt':'jet_phi']
    
    # Assertions for size of matrices
    assert X_train_constit.shape[1] / num_constit_vars == num_max_constits
    assert X_train_track.shape[1] / num_track_vars == num_max_tracks
    assert X_train_MSeg.shape[1] / num_MSeg_vars == num_max_MSegs
    assert X_train_jet.shape[1] == num_jet_vars

    # Split X_test and X_val into track, MSeg, and constit inputs as well
    X_test_constit = X_test.loc[:,'clus_pt_0':'clus_time_'+str(num_max_constits-1)]
    X_test_track = X_test.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
    X_test_MSeg = X_test.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
    X_test_jet = X_test.loc[:,'jet_pt':'jet_phi']

    X_val_constit = X_val.loc[:,'clus_pt_0':'clus_time_'+str(num_max_constits-1)]
    X_val_track = X_val.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
    X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
    X_val_jet = X_val.loc[:,'jet_pt':'jet_phi']

    #Reshape the dataFrames into the shape expected by keras
    #This is an ordered array, so each input is formatted as number of constituents x number of variables
    X_train_constit = X_train_constit.values.reshape(X_train_constit.shape[0],num_max_constits,num_constit_vars)
    print("\nX_train_constit shape: %.0f x %.0f" %  (X_train_constit.shape[1], X_train_constit.shape[2])) 
    print("Number of examples X_train_constit: %.0f" % (X_train_constit.shape[0]))

    X_train_track = X_train_track.values.reshape(X_train_track.shape[0],num_max_tracks,num_track_vars)
    print("X_train_track shape: %.0f x %.0f" %  (X_train_track.shape[1], X_train_track.shape[2]))
    print("Number of examples X_train_track: %.0f" % (X_train_track.shape[0]))
    
    X_train_MSeg = X_train_MSeg.values.reshape(X_train_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)
    print("X_train_MSeg shape: %.0f x %.0f" %  (X_train_MSeg.shape[1], X_train_MSeg.shape[2]))
    print("Number of examples X_train_MSeg: %.0f" % (X_train_MSeg.shape[0]))

    # Testing dataset
    X_test_constit = X_test_constit.values.reshape(X_test_constit.shape[0],num_max_constits,num_constit_vars)
    print("\nX_test_constit shape: %.0f x %.0f" %  (X_test_constit.shape[1], X_train_constit.shape[2]))
    print("Number of examples X_test_constit: %.0f" % (X_test_constit.shape[0]))

    X_test_track = X_test_track.values.reshape(X_test_track.shape[0],num_max_tracks,num_track_vars)
    print("X_test_track shape: %.0f x %.0f" %  (X_test_track.shape[1], X_test_track.shape[2]))
    print("Number of examples X_test_track: %.0f" % (X_test_track.shape[0]))

    X_test_MSeg = X_test_MSeg.values.reshape(X_test_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)
    print("X_test_MSeg shape: %.0f x %.0f" %  (X_test_MSeg.shape[1], X_test_MSeg.shape[2]))
    print("Number of examples X_test_MSeg: %.0f" % (X_test_MSeg.shape[0]))

    # Validation dataset
    X_val_constit = X_val_constit.values.reshape(X_val_constit.shape[0],num_max_constits,num_constit_vars)
    print("\nX_val_constit shape: %.0f x %.0f" %  (X_val_constit.shape[1], X_val_constit.shape[2]))
    print("Number of examples X_val_constit: %.0f" % (X_val_constit.shape[0]))

    X_val_track = X_val_track.values.reshape(X_val_track.shape[0],num_max_tracks,num_track_vars)
    print("X_val_track shape: %.0f x %.0f" %  (X_val_track.shape[1], X_val_track.shape[2]))
    print("Number of examples X_val_track: %.0f" % (X_val_track.shape[0]))

    X_val_MSeg = X_val_MSeg.values.reshape(X_val_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)
    print("X_val_MSeg shape: %.0f x %.0f" %  (X_val_MSeg.shape[1], X_val_MSeg.shape[2]))
    print("Number of examples X_val_MSeg: %.0f" % (X_val_MSeg.shape[0]))

    # Done preparing inputs for model!!
    # Now to setup ML architecture
    # Inputs
    constit_input_tensor = Input(shape=(X_train_constit[0].shape), dtype='float32', name='constit_input')
    # input shape = 3D tensor with shape: (batch, steps, channels)
    # output shape = 3D tensor with shape: (batch, new_steps, filters)
    constit_output_tensor = Conv1D(filters=num_constit_cnn.pop(0), kernel_size=1, activation='relu', input_shape=(X_train_constit[0].shape))(constit_input_tensor)
    for i in range(len(num_constit_cnn)):
       	constit_output_tensor = Conv1D(filters=num_constit_cnn.pop(0), kernel_size=1, activation='relu')(constit_output_tensor)

    # TODO: make helper functions to avoid duplicate code such as below
    track_input_tensor = Input(shape=(X_train_track[0].shape), dtype='float32', name='track_input')
    track_output_tensor = Conv1D(filters=num_track_cnn.pop(0), kernel_size=1, activation='relu', input_shape=(X_train_track[0].shape))(track_input_tensor)
    for i in range(len(num_track_cnn)):
        track_output_tensor = Conv1D(filters=num_track_cnn.pop(0), kernel_size=1, activation='relu')(track_output_tensor)

    MSeg_input_tensor = Input(shape=(X_train_MSeg[0].shape), dtype='float32', name='track_input')
    MSeg_ouput_tensor = Conv1D(filters=num_MSeg_cnn.pop(0), kernel_size=1, activation='relu', input_shape=(X_train_MSeg[0].shape))(MSeg_input_tensor)
    for i in range(len(num_MSeg_cnn)):
        MSeg_output_tensor = Con1D(filters=num_MSeg_cnn.pop(0), kernel_size=1, activation='relu', input_shape=(X_train_MSeg[0].shape))(MSeg_output_tensor)


