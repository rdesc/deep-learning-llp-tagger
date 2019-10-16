import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')

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
Main function
Takes in arguments to change architecture of LSTM network, does training,  then runs evaluate_training to get out training plots and stats

filename: name of input data
frac: fraction of events in filename to use
num_max_constits: how many constituents to input to the constituent LSTM
num_max_tracks: how many tracks to input to track LSTM
num_max_MSegs: how many tracks to input to the muon segment LSTM
num_constit_lstm: how many nodes to include in constituent LSTM
num_track_lstm: how many nodes to include in track LSTM
num_mseg_lstm: how many nodes to include in muon segment LSTM
reg_value: value of regularizer term for LSTM
dropout_value: fraction of Dropout nodes for LSTM and Hidden Layers
epochs: Number of epochs to train for
model_to_do: first part of name to save model by (will add some of these other variables to name)
doTrackLSTM: whether or not to run track LSTM
doMSegLSTM: whether or not to run muon segment LSTM
doParametrization: whether to include mH and mS truth variables in training to enable parametrized training
learning_rate: Starting learning rate for training
'''
def train_llp( filename, frac = 1.0, num_max_constits=30, num_max_tracks=20, num_max_MSegs=70, num_constit_lstm=60, num_track_lstm=60, num_mseg_lstm=25, reg_value=0.001, dropout_value = 0.1,  epochs = 50, model_to_do = "lstm_test" , doTrackLSTM = True, doMSegLSTM = True, doParametrization = False, learning_rate = 0.002):

    #name, filename, frac, num_max_constits, num_max_tracks, num_max_MSegs, num_constit_lstm, num_track_lstm, num_mseg_lstm, reg_value, model_to_do = sys.argv

    #Create a string which will be the name of the model
    #We will use this as directory name to save plots and save the model
    creation_time = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S/'))
    model_to_do = model_to_do+"_fracEvents_" + str(frac) + "_constits_" + str(num_max_constits) + "_tracks_" + str(num_max_tracks) + "_MSegs_" + str(num_max_MSegs) + "_LSTMconstits_" + str(num_constit_lstm) + "_LSTMtracks_" + str(num_track_lstm) + "_LSTMmseg_" + str(num_mseg_lstm) + "_kernelReg_" + str(reg_value) + "_epochs_" + str(epochs) + "_dropout_" + str(dropout_value) + "_doTrackLSTM_" + str(doTrackLSTM) + "_doMSegLSTM_" + str(doMSegLSTM) + "_" + creation_time

    #Make directory
    dirName = "plots/" + model_to_do
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")

    dirName = "plots/trainingDiagrams/" + model_to_do
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")

    dirName = "keras_outputs/" + model_to_do
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")

    destination = "plots/"+model_to_do + "/"
    #Write a file with some details of architecture, will append final stats at end of training
    f = open(destination+"training_details.txt","w+")
    f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (frac, num_max_constits, num_max_tracks, num_max_MSegs, num_constit_lstm, num_track_lstm, num_mseg_lstm, learning_rate) )
    f.close()

    #Convert inputs to correct type
    frac = float(frac)
    reg_value = float(reg_value)
    num_max_constits = int(num_max_constits)
    num_max_tracks = int(num_max_tracks)
    num_max_MSegs = int(num_max_MSegs)
    num_constit_lstm = int(num_constit_lstm)
    num_track_lstm = int(num_track_lstm)
    num_mseg_lstm = int(num_mseg_lstm)
    print(model_to_do)
    learning_rate = float(learning_rate)

   #print("Frac: " + str(frac) + ", max constits: " + str(num_max_constits) + ", max tracks: " + str(num_max_tracks) + ", max MSegs: " + str(num_max_MSegs))

    #Keras magic
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=64)
    tf.set_random_seed(1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)
    
    #Read input pickle
    df = pd.read_pickle(filename)
    #Replace all NaN by 0
    df = df.fillna(0)
    
    
    
    #Delete some 'virtual' variables only needed for pre-processing
    del df['track_sign']
    del df['clus_sign']
    
    #I labelled all truth variables (except parametrization) with aux_*, so remove all those!!!
    auxDelete = [col for col in df if col.startswith("aux")]
    for item in auxDelete:
        del df[item]
    
    #TODO: not leave hardcoded
    #Decides if time is a variable or not
    deleteTime = False
    
    #Delete time variable for clusters and muon segments
    if deleteTime:
        clusTimeDelete = [col for col in df if col.startswith("clusTime")]
        for item in clusTimeDelete:
            del df[item]
    
        segTimeDelete = [col for col in df  if col.startswith("nn_MSeg_t0")]
        for item in segTimeDelete:
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
    X= df.loc[:,'jet_pt':'l4_hcal_29']
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
    
    #model_to_do = "dense"
    
    #Convert labels to categorical (needed for multiclass training) 
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)
    
    #Need to include 'lstm' in input model_to_do to tell code to use LSTM to train
    if("lstm" in model_to_do):
    
        #Hard-coded the number of variables inside each jet constituent, track or muon Segment
        #It is reduced by 1 for constituents and muon segments if time is not included
        num_constit_vars = 12
        if deleteTime == True:
            num_constit_vars = 11
        num_track_vars = 12
        num_MSeg_vars = 5
        if deleteTime == True:
            num_MSeg_vars = 4
        
        num_jet_vars = 3
        
        
        #Construct constituent, track, muon segment and jet dataFrames, dependent on if you include time or not
        #This just chooses the appropriate columns
        #Dependent on correct ordering when extracting and pre-processing!
        if deleteTime:
            X_train_constit = X_train.loc[:,'clus_pt_0':'clus_phi_'+str(num_max_constits-1)]
        else:
            X_train_constit = X_train.loc[:,'clus_pt_0':'clusTime_'+str(num_max_constits-1)]
        
        X_train_track = X_train.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
        if deleteTime:
            X_train_MSeg = X_train.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_'+str(num_max_MSegs-1)]
        else:
            X_train_MSeg = X_train.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
        X_train_jet = X_train.loc[:,'jet_pt':'jet_phi']
        if doParametrization == True:
            X_train_jet = X_train_jet.join(Z_train)

        
        #X_train_jet.join(X_train.loc[:,'llp_mH'])
        #X_train_jet.join(X_train.loc[:,'llp_mS'])
        

        #Does same constructing for test dataset
        if deleteTime:
            X_test_constit = X_test.loc[:,'clus_pt_0':'clus_phi_'+str(num_max_constits-1)]
        else:
            X_test_constit = X_test.loc[:,'clus_pt_0':'clusTime_'+str(num_max_constits-1)]
        X_test_track = X_test.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
        if deleteTime:
            X_test_MSeg = X_test.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_'+str(num_max_MSegs-1)]
        else:
            X_test_MSeg = X_test.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
        X_test_jet = X_test.loc[:,'jet_pt':'jet_phi']
        if doParametrization:
            X_test_jet= X_test_jet.join(Z_test)
        
        #Does same constructing for validation dataset
        if deleteTime:
            X_val_constit = X_val.loc[:,'clus_pt_0':'clus_phi_'+str(num_max_constits-1)]
        else:
            X_val_constit = X_val.loc[:,'clus_pt_0':'clusTime_'+str(num_max_constits-1)]
        X_val_track = X_val.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
        if deleteTime:
            X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_'+str(num_max_MSegs-1)]
        else:
            X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
        X_val_jet = X_val.loc[:,'jet_pt':'jet_phi']
        if doParametrization:
            X_val_jet= X_val_jet.join(Z_val)

        
        #Add the layering constituent information calcualted during pre-processing to right place in constituent dataFrame
        #TODO: move to pre-processing
        for i in range(0,num_max_constits):
            temp_loc = X_train_constit.columns.get_loc('clusTime_'+str(i))
            
            X_train_constit.insert(temp_loc,'l4_hcal_'+str(i),X_train['l4_hcal_'+str(i)])
            X_train_constit.insert(temp_loc,'l3_hcal_'+str(i),X_train['l3_hcal_'+str(i)])
            X_train_constit.insert(temp_loc,'l2_hcal_'+str(i),X_train['l2_hcal_'+str(i)])
            X_train_constit.insert(temp_loc,'l1_hcal_'+str(i),X_train['l1_hcal_'+str(i)])
            X_train_constit.insert(temp_loc,'l4_ecal_'+str(i),X_train['l4_ecal_'+str(i)])
            X_train_constit.insert(temp_loc,'l3_ecal_'+str(i),X_train['l3_ecal_'+str(i)])
            X_train_constit.insert(temp_loc,'l2_ecal_'+str(i),X_train['l2_ecal_'+str(i)])
            X_train_constit.insert(temp_loc,'l1_ecal_'+str(i),X_train['l1_ecal_'+str(i)])
            
            X_test_constit.insert(temp_loc,'l4_hcal_'+str(i),X_test['l4_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l3_hcal_'+str(i),X_test['l3_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l2_hcal_'+str(i),X_test['l2_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l1_hcal_'+str(i),X_test['l1_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l4_ecal_'+str(i),X_test['l4_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l3_ecal_'+str(i),X_test['l3_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l2_ecal_'+str(i),X_test['l2_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l1_ecal_'+str(i),X_test['l1_ecal_'+str(i)])
            
            X_val_constit.insert(temp_loc,'l4_hcal_'+str(i),X_val['l4_hcal_'+str(i)])
            X_val_constit.insert(temp_loc,'l3_hcal_'+str(i),X_val['l3_hcal_'+str(i)])
            X_val_constit.insert(temp_loc,'l2_hcal_'+str(i),X_val['l2_hcal_'+str(i)])
            X_val_constit.insert(temp_loc,'l1_hcal_'+str(i),X_val['l1_hcal_'+str(i)])
            X_val_constit.insert(temp_loc,'l4_ecal_'+str(i),X_val['l4_ecal_'+str(i)])
            X_val_constit.insert(temp_loc,'l3_ecal_'+str(i),X_val['l3_ecal_'+str(i)])
            X_val_constit.insert(temp_loc,'l2_ecal_'+str(i),X_val['l2_ecal_'+str(i)])
            X_val_constit.insert(temp_loc,'l1_ecal_'+str(i),X_val['l1_ecal_'+str(i)])
        
        #Reshape the dataFrames into the shape expected by keras
        #This is an ordered array, so each input is formatted as number of constituents x number of variables
        X_train_constit = X_train_constit.values.reshape(X_train_constit.shape[0],num_max_constits,num_constit_vars)
        X_train_track = X_train_track.values.reshape(X_train_track.shape[0],num_max_tracks,num_track_vars)
        X_train_MSeg = X_train_MSeg.values.reshape(X_train_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)
        
        X_test_constit = X_test_constit.values.reshape(X_test_constit.shape[0],num_max_constits,num_constit_vars)
        X_test_track = X_test_track.values.reshape(X_test_track.shape[0],num_max_tracks,num_track_vars)
        X_test_MSeg = X_test_MSeg.values.reshape(X_test_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)
        
        X_val_constit = X_val_constit.values.reshape(X_val_constit.shape[0],num_max_constits,num_constit_vars)
        X_val_track = X_val_track.values.reshape(X_val_track.shape[0],num_max_tracks,num_track_vars)
        X_val_MSeg = X_val_MSeg.values.reshape(X_val_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)
        
        #Start constructing the network, using keras functional API 

        #Need to tell network shape of input for jet constituents
        constit_input = Input(shape=(X_train_constit[0].shape),dtype='float32',name='constit_input')
        #Have one LSTM layer, with regularizer, tied to input node
        constit_out = LSTM(num_constit_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(constit_input)
        #Have a constit LSTM output, which does the classification using only constituents.
        #This lets you monitor how the consituent LSTM is doing, but is not used for anything else
        constit_output = Dense(3, activation='softmax', name='constit_output')(constit_out)
        
        #Need to tell network shape of input for tracks
        track_input = Input(shape=(X_train_track[0].shape),dtype='float32',name='track_input')
        #Have one LSTM layer, with regularizer, tied to input node
        track_out = LSTM(num_track_lstm , kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(track_input)
        #Have a track LSTM output, which does the classification using only tracks.
        #This lets you monitor how the track LSTM is doing, but is not used for anything else
        track_output = Dense(3, activation='softmax', name='track_output')(track_out)
        
        #Need to tell network shape of input for muon segments
        MSeg_input = Input(shape=(X_train_MSeg[0].shape),dtype='float32',name='MSeg_input')
        #Have one LSTM layer, with regularizer, tied to input node
        MSeg_out = LSTM(num_mseg_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(MSeg_input)
        #Have a muon segment LSTM output, which does the classification using only muon segments.
        #This lets you monitor how the muon segment LSTM is doing, but is not used for anything else
        MSeg_output = Dense(3, activation='softmax', name='MSeg_output')(MSeg_out)
        
        #Have an input for jet variables (pt, eta, phi)
        #Can add parametrization at this point
        jet_input = Input(shape = X_train_jet.values[0].shape, name='jet_input')
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

        #initialize some inputs and output for trainings
        layers_to_input = [constit_input, track_input, MSeg_input, jet_input]
        layers_to_output = [main_output, constit_output, track_output, MSeg_output, jet_output]
        x_to_train = [X_train_constit, X_train_track, X_train_MSeg, X_train_jet.values]
        y_to_train = [y_train, y_train, y_train, y_train, y_train]
        weights_to_train = [weights_train.values, weights_train.values, weights_train.values, weights_train.values, weights_train.values]
        #initialize some inputs and output for validation
        x_to_validate = [X_test_constit, X_test_track, X_test_MSeg, X_test_jet.values]
        y_to_validate = [y_test, y_test, y_test, y_test,y_test]
        weights_to_validate = [weights_test.values, weights_test.values, weights_test.values,weights_test.values,weights_test.values]
        #Weight for each loss function, for main loss. At this point a bit arbitrary
        #TODO: optimise
        weights_for_loss = [1., 0.1, 0.4, 0.2,0.1]
        
        #Change inputs and outputs depending on what variables are being used
        if (doTrackLSTM and not doMSegLSTM):
            layers_to_input = [constit_input, track_input,  jet_input]
            layers_to_output = [main_output, constit_output, track_output, jet_output]
            x_to_train = [X_train_constit, X_train_track, X_train_jet.values]
            y_to_train = [y_train, y_train, y_train,  y_train]
            weights_to_train = [weights_train.values,  weights_train.values, weights_train.values, weights_train.values]
            x_to_validate = [X_test_constit, X_test_track, X_test_jet.values]
            y_to_validate = [y_test, y_test, y_test, y_test]
            weights_to_validate = [weights_test.values,  weights_test.values,weights_test.values,weights_test.values]
            weights_for_loss = [1., 0.1, 0.4, 0.1]
        if (doMSegLSTM and not doTrackLSTM):
            layers_to_input = [constit_input, MSeg_input,  jet_input]
            layers_to_output = [main_output, constit_output, MSeg_output, jet_output]
            x_to_train = [X_train_constit, X_train_MSeg, X_train_jet.values]
            y_to_train = [y_train, y_train, y_train,  y_train]
            weights_to_train = [weights_train.values,  weights_train.values, weights_train.values, weights_train.values]
            x_to_validate = [X_test_constit, X_test_MSeg, X_test_jet.values]
            y_to_validate = [y_test, y_test, y_test, y_test]
            weights_to_validate = [weights_test.values,  weights_test.values,weights_test.values,weights_test.values]
            weights_for_loss = [1., 0.1, 0.2,0.1]
        if (not doTrackLSTM and not doMSegLSTM):
            layers_to_input = [constit_input,  jet_input]
            layers_to_output = [main_output, constit_output, jet_output]
            x_to_train = [X_train_constit, X_train_jet.values]
            y_to_train = [y_train, y_train,  y_train]
            weights_to_train = [weights_train.values, weights_train.values, weights_train.values]
            x_to_validate = [X_test_constit, X_test_jet.values]
            y_to_validate = [y_test, y_test,  y_test]
            weights_to_validate = [weights_test.values,weights_test.values,weights_test.values]
            weights_for_loss = [1., 0.1, 0.1]
        
        #Add everything we just initialised to the model
        model = Model(inputs=layers_to_input, outputs=layers_to_output)
        
        #Make an optimiser. Nadam is good as it has decaying learning rate
        opt = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        #Compile model
        model.compile(optimizer=opt, loss='categorical_crossentropy',
        loss_weights=weights_for_loss, metrics=['accuracy'])
        #Show shape of model in command prompt
        model.summary()
        #Do the training, save it to history for plots
        #Batch size hardcoded to 512, NOT optimised
        #Has EarlyStopping module, if variable under monitor has not gotten better after <patience> epochs, stop, keep best at directory under ModelCheckpoint
        history = model.fit(x_to_train, y_to_train, sample_weight= weights_to_train, epochs=epochs, batch_size=512, validation_data = (x_to_validate, y_to_validate, weights_to_validate),callbacks=[
        EarlyStopping(
        verbose=True,
        patience=10,
        monitor='val_main_output_acc'),
        ModelCheckpoint(
        'keras_outputs/'+model_to_do+'/checkpoint',
        monitor='val_main_output_acc',
        verbose=True,
        save_best_only=True)])
        #Plot accuracy history for training and validation
        plt.clf()
        plt.cla()
        plt.figure()
        plt.plot(history.history['main_output_acc'])
        plt.plot(history.history['val_main_output_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plots/" + model_to_do + "/" + "accuracy_monitoring.pdf", format='pdf', transparent=True)
        plt.clf()
        plt.cla()
        plt.figure()
        # summarize history for loss
        #Plot loss history for training and validation
        plt.plot(history.history['main_output_loss'])
        plt.plot(history.history['val_main_output_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plots/" + model_to_do + "/" + "loss_monitoring.pdf", format='pdf', transparent=True)
        
        #evaluate_model makes plots, like ROC curve, gets stats about training
        evaluate_model(X_val, y_val, weights_val, mcWeights_val, Z_val, model_to_do, deleteTime, num_constit_lstm, num_track_lstm, num_mseg_lstm, reg_value, doTrackLSTM, doMSegLSTM, doParametrization, learning_rate)
	
	
    #Dense network, much simpler than above LSTM 
    if ("dense" in model_to_do):

        if deleteTime:
            num_constit_vars = 11
        else:
            num_constit_vars = 12
        num_track_vars = 12
        if deleteTime:
            num_MSeg_vars = 4
        else:
            num_MSeg_vars = 5
        
        num_jet_vars = 3
        
        
        if deleteTime:
            X_train_constit = X_train.loc[:,'clus_pt_0':'clus_phi_'+str(num_max_constits-1)]
        else:
            X_train_constit = X_train.loc[:,'clus_pt_0':'clusTime_'+str(num_max_constits-1)]
        
        X_train_track = X_train.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
        if deleteTime:
            X_train_MSeg = X_train.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_'+str(num_max_MSegs-1)]
        else:
            X_train_MSeg = X_train.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
        X_train_jet = X_train.loc[:,'jet_pt':'jet_phi']
        if doParametrization:
            X_train_jet.join(Z_train)
        
        #X_train_jet.join(X_train.loc[:,'llp_mH'])
        #X_train_jet.join(X_train.loc[:,'llp_mS'])
        
        if deleteTime:
            X_test_constit = X_test.loc[:,'clus_pt_0':'clus_phi_'+str(num_max_constits-1)]
        else:
            X_test_constit = X_test.loc[:,'clus_pt_0':'clusTime_'+str(num_max_constits-1)]
        X_test_track = X_test.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
        if deleteTime:
            X_test_MSeg = X_test.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_'+str(num_max_MSegs-1)]
        else:
            X_test_MSeg = X_test.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
        X_test_jet = X_test.loc[:,'jet_pt':'jet_phi']
        
        if deleteTime:
            X_val_constit = X_val.loc[:,'clus_pt_0':'clus_phi_'+str(num_max_constits-1)]
        else:
            X_val_constit = X_val.loc[:,'clus_pt_0':'clusTime_'+str(num_max_constits-1)]
        X_val_track = X_val.loc[:,'nn_track_pt_0':'nn_track_SCTHits_'+str(num_max_tracks-1)]
        if deleteTime:
            X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_'+str(num_max_MSegs-1)]
        else:
            X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_'+str(num_max_MSegs-1)]
        X_val_jet = X_val.loc[:,'jet_pt':'jet_phi']
        if doParametrization:
            X_val_jet.join(Z_val)
        
        for i in range(0,num_max_constits):
            temp_loc = X_train_constit.columns.get_loc('clusTime_'+str(i))
            
            X_train_constit.insert(temp_loc,'l4_hcal_'+str(i),X_train['l4_hcal_'+str(i)])
            X_train_constit.insert(temp_loc,'l3_hcal_'+str(i),X_train['l3_hcal_'+str(i)])
            X_train_constit.insert(temp_loc,'l2_hcal_'+str(i),X_train['l2_hcal_'+str(i)])
            X_train_constit.insert(temp_loc,'l1_hcal_'+str(i),X_train['l1_hcal_'+str(i)])
            X_train_constit.insert(temp_loc,'l4_ecal_'+str(i),X_train['l4_ecal_'+str(i)])
            X_train_constit.insert(temp_loc,'l3_ecal_'+str(i),X_train['l3_ecal_'+str(i)])
            X_train_constit.insert(temp_loc,'l2_ecal_'+str(i),X_train['l2_ecal_'+str(i)])
            X_train_constit.insert(temp_loc,'l1_ecal_'+str(i),X_train['l1_ecal_'+str(i)])
            
            X_test_constit.insert(temp_loc,'l4_hcal_'+str(i),X_test['l4_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l3_hcal_'+str(i),X_test['l3_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l2_hcal_'+str(i),X_test['l2_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l1_hcal_'+str(i),X_test['l1_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l4_ecal_'+str(i),X_test['l4_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l3_ecal_'+str(i),X_test['l3_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l2_ecal_'+str(i),X_test['l2_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l1_ecal_'+str(i),X_test['l1_ecal_'+str(i)])
            
            X_val_constit.insert(temp_loc,'l4_hcal_'+str(i),X_val['l4_hcal_'+str(i)])
            X_val_constit.insert(temp_loc,'l3_hcal_'+str(i),X_val['l3_hcal_'+str(i)])
            X_val_constit.insert(temp_loc,'l2_hcal_'+str(i),X_val['l2_hcal_'+str(i)])
            X_val_constit.insert(temp_loc,'l1_hcal_'+str(i),X_val['l1_hcal_'+str(i)])
            X_val_constit.insert(temp_loc,'l4_ecal_'+str(i),X_val['l4_ecal_'+str(i)])
            X_val_constit.insert(temp_loc,'l3_ecal_'+str(i),X_val['l3_ecal_'+str(i)])
            X_val_constit.insert(temp_loc,'l2_ecal_'+str(i),X_val['l2_ecal_'+str(i)])
            X_val_constit.insert(temp_loc,'l1_ecal_'+str(i),X_val['l1_ecal_'+str(i)])




        X_train = pd.concat([X_train_constit, X_train_track, X_train_MSeg, X_train_jet], axis=1, sort=False)
        X_test = pd.concat([X_test_constit, X_test_track, X_test_MSeg, X_test_jet], axis=1, sort=False)
        X_val = pd.concat([X_val_constit, X_val_track, X_val_MSeg, X_val_jet], axis=1, sort=False)
        
        
        model = Sequential()
        
        model.add(Dense(2000, input_dim=X_train.shape[1]))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(1000))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train.values, y_train, sample_weight= weights_train.values, epochs=epochs, batch_size=512, validation_data = (X_test.values, y_test, weights_test.values),callbacks=[
        EarlyStopping(
        verbose=True,
        patience=20,
        monitor='val_acc'),
        ModelCheckpoint(
        'keras_outputs/'+model_to_do+'/checkpoint',
        monitor='val_acc',
        verbose=True,
        save_best_only=True)])
        plt.clf()
        plt.cla()
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plots/"  + model_to_do + "/" + "accuracy_monitoring.pdf", format='pdf', transparent=True)
        plt.clf()
        plt.cla()
        plt.figure()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("plots/" + model_to_do + "/" + "loss_monitoring.pdf", format='pdf', transparent=True)
        evaluate_model(X_val, y_val, weights_val, mcWeights_val, Z_val, model_to_do, deleteTime, num_constit_lstm, num_track_lstm, num_mseg_lstm, reg_value, doTrackLSTM, doMSegLSTM, doParametrization)
