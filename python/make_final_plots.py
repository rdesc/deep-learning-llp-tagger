import numpy as np
import seaborn as sns

import pandas as pd


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
import matplotlib.colors as mcolorso

import sklearn
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Highway, Dropout, Masking, LSTM, SimpleRNN, Input, CuDNNLSTM
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.regularizers import l1, l2, L1L2

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def plot_three_histos(signal,qcd,bib,name,xmin,xmax,bins, signal_right, qcd_right, bib_right, model_to_do):
    fig,ax = plt.subplots()

    print(name)


    sns.regplot(x=signal, y=signal_right, x_bins=bins, fit_reg=None, color='red', ax=ax, label="Signal")
    sns.regplot(x=qcd, y=qcd_right, x_bins=bins, fit_reg=None, color='blue', ax=ax, label="QCD")
    sns.regplot(x=bib, y=bib_right, x_bins=bins, fit_reg=None, color='green', ax=ax, label="BIB")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel(name)


    plt.savefig("plots/" + model_to_do + "/" +name+ ".png", format='png', transparent=False)
    plt.clf()
    plt.close()

def plot_truth_histos(signal,name,bins, signal_right, model_to_do):
    fig,ax = plt.subplots()

    print(signal.shape)
    print(signal)
    print(signal_right.shape)
    print(signal_right)
    print(name)


    sns.regplot(x=signal, y=signal_right, x_bins=bins, fit_reg=None, color='red', ax=ax, label="Signal")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel(name)


    plt.savefig("plots/" + model_to_do + "/" +name+ ".png", format='png', transparent=False)
    plt.clf()
    plt.close()

def do_plotting(signal,qcd,bib,name,xmin,xmax,bins, signal_right, qcd_right, bib_right, model_to_do):

    filter_nn_MSeg = [col for col in signal if col == name]

    #signal_MSeg = remove_values_from_list(signal[filter_nn_MSeg].values.flatten(),np.nan)
    #qcd_MSeg = remove_values_from_list(qcd[filter_nn_MSeg].values.flatten(),np.nan)
    #bib_MSeg = remove_values_from_list(bib[filter_nn_MSeg].values.flatten(),np.nan)

    signal_MSeg = signal[filter_nn_MSeg].dropna()
    qcd_MSeg = qcd[filter_nn_MSeg].dropna()
    bib_MSeg = bib[filter_nn_MSeg].dropna()


    plot_three_histos(signal_MSeg.values.flatten(),qcd_MSeg.values.flatten(),bib_MSeg.values.flatten(),name,xmin,xmax,bins, signal_right, qcd_right, bib_right, model_to_do)


def do_plotting_truth(signal,name,bins, signal_right, model_to_do):

    filter_nn_MSeg = [col for col in signal if col == name]

    #signal_MSeg = remove_values_from_list(signal[filter_nn_MSeg].values.flatten(),np.nan)
    #qcd_MSeg = remove_values_from_list(qcd[filter_nn_MSeg].values.flatten(),np.nan)
    #bib_MSeg = remove_values_from_list(bib[filter_nn_MSeg].values.flatten(),np.nan)

    signal_MSeg = signal[filter_nn_MSeg].dropna()


    plot_truth_histos(signal_MSeg.values.flatten(),name,bins, signal_right, model_to_do)


def plot_vars_final(file_name, model_to_do, doParametrization, deleteTime, doTrackLSTM = True, doMSegLSTM = True, num_max_constits=30, num_max_tracks=20, num_max_MSegs=70, num_constit_lstm=60, num_track_lstm=60, num_mseg_lstm=25, reg_value=0.001, dropout_value = 0.1):

    df = pd.read_pickle(file_name)
    df = df.fillna(0)



    del df['track_sign']
    del df['clus_sign']

    deleteTime = False

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


    Y = df['label']
    weights = df['flatWeight']
    mcWeights = df['mcEventWeight']
    X= df.loc[:,'jet_pt':'l4_hcal_29']
    del X['llp_mS']
    del X['llp_mH']
    del X['jet_isClean_LooseBadLLP']
    #Z = df.loc[:,'llp_mH':'llp_mS']
    Z = df.loc[:,'eventNumber':'runNumber']
    Z = Z.join(df.loc[:,'jet_pt'])

    




    X_train, X_test, y_train, y_test, weights_train, weights_test, mcWeights_train, mcWeights_test,  Z_train, Z_test = train_test_split(X, Y, weights, mcWeights, Z, test_size = 0.99)
    del X_train
    del y_train
    del weights_train
    del mcWeights_train
    del Z_train
    del X
    del Y
    del Z


    X_test, X_val, y_test, y_val, weights_test, weights_val, mcWeights_test, mcWeights_val, Z_test, Z_val = train_test_split(X_test, y_test, weights_test, mcWeights_test,  Z_test, test_size = 0.99)

    del X_test
    del y_test
    del weights_test
    del mcWeights_test
    del Z_test

    
    signal = X_val[y_val == 1]
    print(signal.shape[0])
    qcd = X_val[y_val == 0]
    print(qcd.shape[0])
    bib = X_val[y_val == 2]
    print(bib.shape[0])


    num_jet_vars = 3


    num_constit_vars = 12
    if deleteTime == True:
        num_constit_vars = 11
    num_track_vars = 12
    num_MSeg_vars = 5
    if deleteTime == True:
        num_MSeg_vars = 4


    num_max_constits = 30
    num_max_tracks = 20
    num_max_MSegs = 70

    if deleteTime:
        X_val_constit = X_val.loc[:,'clus_pt_0':'clus_phi_29']
    else:
        X_val_constit = X_val.loc[:,'clus_pt_0':'clusTime_29']
    X_val_track = X_val.loc[:,'nn_track_pt_0':'nn_track_SCTHits_19']
    if deleteTime:
        X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_69']
    else:
        X_val_MSeg = X_val.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_69']
    X_val_jet = X_val.loc[:,'jet_pt':'jet_phi']
    if doParametrization:
        X_val_jet = X_val_jet.join(Z_val)

    for i in range(0,num_max_constits):
        temp_loc = X_val_constit.columns.get_loc('clus_phi_'+str(i))

        X_val_constit.insert(temp_loc,'l4_hcal_'+str(i),X_val['l4_hcal_'+str(i)])
        X_val_constit.insert(temp_loc,'l3_hcal_'+str(i),X_val['l3_hcal_'+str(i)])
        X_val_constit.insert(temp_loc,'l2_hcal_'+str(i),X_val['l2_hcal_'+str(i)])
        X_val_constit.insert(temp_loc,'l1_hcal_'+str(i),X_val['l1_hcal_'+str(i)])
        X_val_constit.insert(temp_loc,'l4_ecal_'+str(i),X_val['l4_ecal_'+str(i)])
        X_val_constit.insert(temp_loc,'l3_ecal_'+str(i),X_val['l3_ecal_'+str(i)])
        X_val_constit.insert(temp_loc,'l2_ecal_'+str(i),X_val['l2_ecal_'+str(i)])
        X_val_constit.insert(temp_loc,'l1_ecal_'+str(i),X_val['l1_ecal_'+str(i)])

    print(list(X_val_constit.columns))
    print(list(X_val_track.columns))
    print(list(X_val_MSeg.columns))
    print(list(X_val_jet.columns))

    X_val_constit_copy = X_val_constit.copy()
    X_val_mseg_copy = X_val_MSeg.copy()

    X_val_constit = X_val_constit.values.reshape(X_val_constit.shape[0],num_max_constits,num_constit_vars)
    X_val_track = X_val_track.values.reshape(X_val_track.shape[0],num_max_tracks,num_track_vars)
    X_val_MSeg = X_val_MSeg.values.reshape(X_val_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)



    constit_input = Input(shape=(X_val_constit[0].shape),dtype='float32',name='constit_input')
    constit_out = LSTM(num_constit_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(constit_input)
    constit_output = Dense(3, activation='softmax', name='constit_output')(constit_out)

    track_input = Input(shape=(X_val_track[0].shape),dtype='float32',name='track_input')
    track_out = LSTM(num_track_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(track_input)
    track_output = Dense(3, activation='softmax', name='track_output')(track_out)

    MSeg_input = Input(shape=(X_val_MSeg[0].shape),dtype='float32',name='MSeg_input')
    MSeg_out = LSTM(num_mseg_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(MSeg_input)
    MSeg_output = Dense(3, activation='softmax', name='MSeg_output')(MSeg_out)

    jet_input = Input(shape = X_val_jet.values[0].shape, name='jet_input')
    jet_out = Dense(3)(jet_input)
    jet_output = Dense(3, activation='softmax', name='jet_output')(jet_out)

    layersToConcatenate = [constit_out, track_out, MSeg_out, jet_input]

    if (doTrackLSTM and not doMSegLSTM):
        layersToConcatenate = [constit_out, track_out, jet_input]
    if (doMSegLSTM and not doTrackLSTM):
        layersToConcatenate = [constit_out, MSeg_out, jet_input]
    if (not doTrackLSTM and not doMSegLSTM):
        layersToConcatenate = [constit_out, jet_input]


    x = keras.layers.concatenate(layersToConcatenate)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    main_output = Dense(3, activation='softmax', name='main_output')(x)

    layers_to_input = [constit_input, track_input, MSeg_input, jet_input]
    layers_to_output = [main_output, constit_output, track_output, MSeg_output, jet_output]
    x_to_validate = [X_val_constit, X_val_track, X_val_MSeg, X_val_jet.values]
    y_to_validate = [y_val, y_val, y_val, y_val,y_val]
    weights_to_validate = [weights_val.values, weights_val.values, weights_val.values,weights_val.values,weights_val.values]
    weights_for_loss = [1., 0.1, 0.4, 0.2,0.1]


    if (doTrackLSTM and not doMSegLSTM):
        layers_to_input = [constit_input, track_input,  jet_input]
        layers_to_output = [main_output, constit_output, track_output, jet_output]
        x_to_validate = [X_val_constit, X_val_track, X_val_jet.values]
        y_to_validate = [y_val, y_val, y_val, y_val]
        weights_to_validate = [weights_val.values,  weights_val.values,weights_val.values,weights_val.values]
        weights_for_loss = [1., 0.1, 0.4, 0.1]
    if (doMSegLSTM and not doTrackLSTM):
        print("HERE")
        layers_to_input = [constit_input, MSeg_input,  jet_input]
        layers_to_output = [main_output, constit_output, MSeg_output, jet_output]
        x_to_validate = [X_val_constit, X_val_MSeg, X_val_jet.values]
        y_to_validate = [y_val, y_val, y_val, y_val]
        weights_to_validate = [weights_val.values,  weights_val.values,weights_val.values,weights_val.values]
        weights_for_loss = [1., 0.1, 0.2,0.1]
    if (not doTrackLSTM and not doMSegLSTM):
        layers_to_input = [constit_input,  jet_input]
        layers_to_output = [main_output, constit_output, jet_output]
        x_to_validate = [X_val_constit, X_val_jet.values]
        y_to_validate = [y_val, y_val,  y_val]
        weights_to_validate = [weights_val.values,weights_val.values,weights_val.values]
        weights_for_loss = [1., 0.1, 0.1]

    model = Model(inputs=layers_to_input, outputs=layers_to_output)


    #rmsprop = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
        loss_weights=weights_for_loss, metrics=['accuracy'])
    model.load_weights("keras_outputs/"+model_to_do+'/checkpoint')

    # get the architecture as a json string
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open('keras_outputs/architecture.json', 'w') as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights('keras_outputs/weights.h5')

    if ("dense" in model_to_do):
        prediction = model.predict(X_val.values, verbose=True)

    elif ("lstm" in model_to_do):
        prediction = model.predict(x_to_validate, verbose = True)
        validation_prediction = prediction
        prediction = prediction[0]

    f = open("keras_validation_sep17.txt","w+")


    for (row,pred_main, pred_clus, pred_track, pred_mseg) in zip(Z_val.itertuples(index=True, name='Pandas'), validation_prediction[0], validation_prediction[1], validation_prediction[2], validation_prediction[3]):
        f.write("%.0f,%.0f,%.6f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f\n" % (getattr(row, "runNumber"), getattr(row, "eventNumber"), getattr(row, "jet_pt"), pred_main[0], pred_main[1], pred_main[2], pred_clus[0], pred_clus[1], pred_clus[2], pred_track[0], pred_track[1], pred_track[2], pred_mseg[0], pred_mseg[1], pred_mseg[2]) )


    f.close()

    '''
    f_clus = open("clus_keras_validation.txt","w+")

    print(X_val_constit_copy)
    print(Z_val)

    for (row,constit_test) in zip(Z_val.itertuples(index=True, name='Pandas'), X_val_constit_copy.itertuples(index=True, name='Pandas')):
        print(constit_test)
        f_clus.write("%.0f,%.0f,%.6f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f\n" % (getattr(row, "runNumber"), getattr(row, "eventNumber"), getattr(row, "jet_pt"), getattr(constit_test,"clus_pt_0"), getattr(constit_test,"clus_eta_0"), getattr(constit_test,"clus_phi_0"), getattr(constit_test,"clus_l1ecal_0"), getattr(constit_test,"clus_l2ecal_0"), getattr(constit_test,"clus_l3ecal_0"), getattr(constit_test,"clus_l4ecal_0"), getattr(constit_test,"clus_l1hcal_0"), getattr(constit_test,"clus_l2hcal_0"), getattr(constit_test,"clus_l3hcal_0"), getattr(constit_test,"clus_l4hcal_0"), getattr(constit_test,"clusTime_0"), getattr(constit_test,"clus_pt_1"), getattr(constit_test,"clus_eta_1"), getattr(constit_test,"clus_phi_1"), getattr(constit_test,"clus_l1ecal_1"), getattr(constit_test,"clus_l2ecal_1"), getattr(constit_test,"clus_l3ecal_1"), getattr(constit_test,"clus_l4ecal_1"), getattr(constit_test,"clus_l1hcal_1"), getattr(constit_test,"clus_l2hcal_1"), getattr(constit_test,"clus_l3hcal_1"), getattr(constit_test,"clus_l4hcal_1"), getattr(constit_test,"clusTime_1"), getattr(constit_test,"clus_pt_2"), getattr(constit_test,"clus_eta_2"), getattr(constit_test,"clus_phi_2"), getattr(constit_test,"clus_l1ecal_2"), getattr(constit_test,"clus_l2ecal_2"), getattr(constit_test,"clus_l3ecal_2"), getattr(constit_test,"clus_l4ecal_2"), getattr(constit_test,"clus_l1hcal_2"), getattr(constit_test,"clus_l2hcal_2"), getattr(constit_test,"clus_l3hcal_2"), getattr(constit_test,"clus_l4hcal_2"), getattr(constit_test,"clusTime_2") ) )


    f_clus.close()
    '''

    f_mseg = open("mseg_keras_validation.txt","w+")

    print(X_val_mseg_copy)
    print(Z_val)

    for (row,nn_MSeg_test) in zip(Z_val.itertuples(index=True, name='Pandas'), X_val_mseg_copy.itertuples(index=True, name='Pandas')):
        f_mseg.write("%.0f,%.0f,%.6f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f,%.7f\n" % (getattr(row, "runNumber"), getattr(row, "eventNumber"), getattr(row, "jet_pt"), nn_MSeg_test[1], nn_MSeg_test[2], nn_MSeg_test[3], nn_MSeg_test[4], nn_MSeg_test[5], nn_MSeg_test[6], nn_MSeg_test[7], nn_MSeg_test[8], nn_MSeg_test[9], nn_MSeg_test[10], nn_MSeg_test[11], nn_MSeg_test[12], nn_MSeg_test[13], nn_MSeg_test[14], nn_MSeg_test[15] ) )


    f_mseg.close()


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


    print("Preparing plotting script")
    xmin_dict = {"jet_pt":0, "jet_eta":-1, "jet_phi":-1, "clus_pt_0":0,"clus_pt_1":0,"clus_pt_2":0,"clus_pt_3":0,"clus_pt_4":0,"clus_eta_1":-1,"clus_phi_1":-1,"clus_eta_2":-1,"clus_phi_2":-1,"clus_eta_3":-1,"clus_phi_3":-1,"clus_eta_4":-1,"clus_phi_4":-1,"clusTime_0":-10,"clusTime_1":-10,"clusTime_2":-10,"clusTime_3":-10,"clusTime_4":-10,"nn_track_pt_0":0,"nn_track_eta_0":-1,"nn_track_phi_0":-1,"nn_track_d0_0":0,"nn_track_z0_0":0,"nn_track_PixelShared_0":-1,"nn_track_PixelSplit_0":-1,"nn_track_SCTShared_0":-1,"nn_track_PixelHoles_0":-1,"nn_track_SCTHoles_0":-1,"nn_track_PixelHits_0":-1,"nn_track_SCTHits_0":-1,"nn_track_pt_1":0,"nn_track_eta_1":-1,"nn_track_phi_1":-1,"nn_track_d0_1":0,"nn_track_z0_1":0,"nn_track_PixelShared_1":-1,"nn_track_PixelSplit_1":-1,"nn_track_SCTShared_1":-1,"nn_track_PixelHoles_1":-1,"nn_track_SCTHoles_1":-1,"nn_track_PixelHits_1":-1,"nn_track_SCTHits_1":-1,"nn_track_pt_2":0,"nn_track_eta_2":-1,"nn_track_phi_2":-1,"nn_track_d0_2":0,"nn_track_z0_2":0,"nn_track_PixelShared_2":-1,"nn_track_PixelSplit_2":-1,"nn_track_SCTShared_2":-1,"nn_track_PixelHoles_2":-1,"nn_track_SCTHoles_2":-1,"nn_track_PixelHits_2":-1,"nn_track_SCTHits_2":-1,"nn_track_pt_3":0,"nn_track_eta_3":-1,"nn_track_phi_3":-1,"nn_track_d0_3":0,"nn_track_z0_3":0,"nn_track_PixelShared_3":-1,"nn_track_PixelSplit_3":-1,"nn_track_SCTShared_3":-1,"nn_track_PixelHoles_3":-1,"nn_track_SCTHoles_3":-1,"nn_track_PixelHits_3":-1,"nn_track_SCTHits_0":-1,"nn_track_pt_4":0,"nn_track_eta_4":-1,"nn_track_phi_4":-1,"nn_track_d0_4":0,"nn_track_z0_4":0,"nn_track_PixelShared_4":-1,"nn_track_PixelSplit_4":-1,"nn_track_SCTShared_4":-1,"nn_track_PixelHoles_4":-1,"nn_track_SCTHoles_4":-1,"nn_track_PixelHits_4":-1,"nn_track_SCTHits_4":-1,"nn_MSeg_etaPos_0":-1,"nn_MSeg_phiPos_0":-1,"nn_MSeg_etaDir_0":-8,"nn_MSeg_phiDir_0":-1,"nn_MSeg_t0_0":-10,"nn_MSeg_etaPos_1":-1,"nn_MSeg_phiPos_1":-1,"nn_MSeg_etaDir_1":-8,"nn_MSeg_phiDir_1":-1,"nn_MSeg_t0_1":-10,"nn_MSeg_etaPos_2":-1,"nn_MSeg_phiPos_2":-1,"nn_MSeg_etaDir_2":-8,"nn_MSeg_phiDir_2":-1,"nn_MSeg_t0_2":-10,"nn_MSeg_etaPos_3":-1,"nn_MSeg_phiPos_3":-1,"nn_MSeg_etaDir_3":-8,"nn_MSeg_phiDir_3":-1,"nn_MSeg_t0_3":-10,"nn_MSeg_etaPos_4":-1,"nn_MSeg_phiPos_4":-1,"nn_MSeg_etaDir_4":-8,"nn_MSeg_phiDir_4":-1,"nn_MSeg_t0_4":-10,"l1_ecal_0":0,"l2_ecal_0":0,"l3_ecal_0":0,"l4_ecal_0":0,"l1_hcal_0":0,"l2_hcal_0":0,"l3_hcal_0":0,"l4_hcal_0":0,"l1_ecal_1":0,"l2_ecal_1":0,"l3_ecal_1":0,"l4_ecal_1":0,"l1_hcal_1":0,"l2_hcal_1":0,"l3_hcal_1":0,"l4_hcal_1":0,"l1_ecal_2":0,"l2_ecal_2":0,"l3_ecal_2":0,"l4_ecal_2":0,"l1_hcal_2":0,"l2_hcal_2":0,"l3_hcal_2":0,"l4_hcal_2":0,"l1_ecal_2":0,"l2_ecal_3":0,"l3_ecal_3":0,"l4_ecal_3":0,"l1_hcal_3":0,"l2_hcal_3":0,"l3_hcal_3":0,"l4_hcal_3":0,"l1_ecal_4":0,"l2_ecal_4":0,"l3_ecal_4":0,"l4_ecal_4":0,"l1_hcal_4":0,"l2_hcal_4":0,"l3_hcal_4":0,"l4_hcal_4":0}


    xmax_dict = {"jet_pt":0, "jet_eta":-1, "jet_phi":-1, "clus_pt_0":0,"clus_pt_1":0,"clus_pt_2":0,"clus_pt_3":0,"clus_pt_4":0,"clus_eta_1":-1,"clus_phi_1":-1,"clus_eta_2":-1,"clus_phi_2":-1,"clus_eta_3":-1,"clus_phi_3":-1,"clus_eta_4":-1,"clus_phi_4":-1,"clusTime_0":-10,"clusTime_1":-10,"clusTime_2":-10,"clusTime_3":-10,"clusTime_4":-10,"nn_track_pt_0":0,"nn_track_eta_0":-1,"nn_track_phi_0":-1,"nn_track_d0_0":0,"nn_track_z0_0":0,"nn_track_PixelShared_0":-1,"nn_track_PixelSplit_0":-1,"nn_track_SCTShared_0":-1,"nn_track_PixelHoles_0":-1,"nn_track_SCTHoles_0":-1,"nn_track_PixelHits_0":-1,"nn_track_SCTHits_0":-1,"nn_track_pt_1":0,"nn_track_eta_1":-1,"nn_track_phi_1":-1,"nn_track_d0_1":0,"nn_track_z0_1":0,"nn_track_PixelShared_1":-1,"nn_track_PixelSplit_1":-1,"nn_track_SCTShared_1":-1,"nn_track_PixelHoles_1":-1,"nn_track_SCTHoles_1":-1,"nn_track_PixelHits_1":-1,"nn_track_SCTHits_1":-1,"nn_track_pt_2":0,"nn_track_eta_2":-1,"nn_track_phi_2":-1,"nn_track_d0_2":0,"nn_track_z0_2":0,"nn_track_PixelShared_2":-1,"nn_track_PixelSplit_2":-1,"nn_track_SCTShared_2":-1,"nn_track_PixelHoles_2":-1,"nn_track_SCTHoles_2":-1,"nn_track_PixelHits_2":-1,"nn_track_SCTHits_2":-1,"nn_track_pt_3":0,"nn_track_eta_3":-1,"nn_track_phi_3":-1,"nn_track_d0_3":0,"nn_track_z0_3":0,"nn_track_PixelShared_3":-1,"nn_track_PixelSplit_3":-1,"nn_track_SCTShared_3":-1,"nn_track_PixelHoles_3":-1,"nn_track_SCTHoles_3":-1,"nn_track_PixelHits_3":-1,"nn_track_SCTHits_0":-1,"nn_track_pt_4":0,"nn_track_eta_4":-1,"nn_track_phi_4":-1,"nn_track_d0_4":0,"nn_track_z0_4":0,"nn_track_PixelShared_4":-1,"nn_track_PixelSplit_4":-1,"nn_track_SCTShared_4":-1,"nn_track_PixelHoles_4":-1,"nn_track_SCTHoles_4":-1,"nn_track_PixelHits_4":-1,"nn_track_SCTHits_4":-1,"nn_MSeg_etaPos_0":-1,"nn_MSeg_phiPos_0":-1,"nn_MSeg_etaDir_0":-8,"nn_MSeg_phiDir_0":-1,"nn_MSeg_t0_0":-10,"nn_MSeg_etaPos_1":-1,"nn_MSeg_phiPos_1":-1,"nn_MSeg_etaDir_1":-8,"nn_MSeg_phiDir_1":-1,"nn_MSeg_t0_1":-10,"nn_MSeg_etaPos_2":-1,"nn_MSeg_phiPos_2":-1,"nn_MSeg_etaDir_2":-8,"nn_MSeg_phiDir_2":-1,"nn_MSeg_t0_2":-10,"nn_MSeg_etaPos_3":-1,"nn_MSeg_phiPos_3":-1,"nn_MSeg_etaDir_3":-8,"nn_MSeg_phiDir_3":-1,"nn_MSeg_t0_3":-10,"nn_MSeg_etaPos_4":-1,"nn_MSeg_phiPos_4":-1,"nn_MSeg_etaDir_4":-8,"nn_MSeg_phiDir_4":-1,"nn_MSeg_t0_4":-10,"l1_ecal_0":0,"l2_ecal_0":0,"l3_ecal_0":0,"l4_ecal_0":0,"l1_hcal_0":0,"l2_hcal_0":0,"l3_hcal_0":0,"l4_hcal_0":0,"l1_ecal_1":0,"l2_ecal_1":0,"l3_ecal_1":0,"l4_ecal_1":0,"l1_hcal_1":0,"l2_hcal_1":0,"l3_hcal_1":0,"l4_hcal_1":0,"l1_ecal_2":0,"l2_ecal_2":0,"l3_ecal_2":0,"l4_ecal_2":0,"l1_hcal_2":0,"l2_hcal_2":0,"l3_hcal_2":0,"l4_hcal_2":0,"l1_ecal_2":0,"l2_ecal_3":0,"l3_ecal_3":0,"l4_ecal_3":0,"l1_hcal_3":0,"l2_hcal_3":0,"l3_hcal_3":0,"l4_hcal_3":0,"l1_ecal_4":0,"l2_ecal_4":0,"l3_ecal_4":0,"l4_ecal_4":0,"l1_hcal_4":0,"l2_hcal_4":0,"l3_hcal_4":0,"l4_hcal_4":0}


    bin_dict = {"jet_pt":20, "jet_eta":20, "jet_phi":20, "clus_pt_0":20,"clus_pt_1":20,"clus_pt_2":20,"clus_pt_3":20,"clus_pt_4":20,"clus_eta_1":20,"clus_phi_1":20,"clus_eta_2":20,"clus_phi_2":20,"clus_eta_3":20,"clus_phi_3":20,"clus_eta_4":20,"clus_phi_4":20,"clusTime_0":200,"clusTime_1":200,"clusTime_2":200,"clusTime_3":200,"clusTime_4":200,"nn_track_pt_0":20,"nn_track_eta_0":20,"nn_track_phi_0":20,"nn_track_d0_0":20,"nn_track_z0_0":20,"nn_track_PixelShared_0":20,"nn_track_PixelSplit_0":20,"nn_track_SCTShared_0":20,"nn_track_PixelHoles_0":20,"nn_track_SCTHoles_0":20,"nn_track_PixelHits_0":20,"nn_track_SCTHits_0":20,"nn_track_pt_1":20,"nn_track_eta_1":20,"nn_track_phi_1":20,"nn_track_d0_1":20,"nn_track_z0_1":20,"nn_track_PixelShared_1":20,"nn_track_PixelSplit_1":20,"nn_track_SCTShared_1":20,"nn_track_PixelHoles_1":20,"nn_track_SCTHoles_1":20,"nn_track_PixelHits_1":20,"nn_track_SCTHits_1":20,"nn_track_pt_2":20,"nn_track_eta_2":20,"nn_track_phi_2":20,"nn_track_d0_2":20,"nn_track_z0_2":20,"nn_track_PixelShared_2":20,"nn_track_PixelSplit_2":20,"nn_track_SCTShared_2":20,"nn_track_PixelHoles_2":20,"nn_track_SCTHoles_2":20,"nn_track_PixelHits_2":20,"nn_track_SCTHits_2":20,"nn_track_pt_3":20,"nn_track_eta_3":20,"nn_track_phi_3":20,"nn_track_d0_3":20,"nn_track_z0_3":20,"nn_track_PixelShared_3":20,"nn_track_PixelSplit_3":20,"nn_track_SCTShared_3":20,"nn_track_PixelHoles_3":20,"nn_track_SCTHoles_3":20,"nn_track_PixelHits_3":20,"nn_track_SCTHits_0":20,"nn_track_pt_4":20,"nn_track_eta_4":20,"nn_track_phi_4":20,"nn_track_d0_4":20,"nn_track_z0_4":20,"nn_track_PixelShared_4":20,"nn_track_PixelSplit_4":20,"nn_track_SCTShared_4":20,"nn_track_PixelHoles_4":20,"nn_track_SCTHoles_4":20,"nn_track_PixelHits_4":20,"nn_track_SCTHits_4":20,"nn_MSeg_etaPos_0":20,"nn_MSeg_phiPos_0":20,"nn_MSeg_etaDir_0":20,"nn_MSeg_phiDir_0":20,"nn_MSeg_t0_0":200,"nn_MSeg_etaPos_1":20,"nn_MSeg_phiPos_1":20,"nn_MSeg_etaDir_1":20,"nn_MSeg_phiDir_1":20,"nn_MSeg_t0_1":200,"nn_MSeg_etaPos_2":20,"nn_MSeg_phiPos_2":20,"nn_MSeg_etaDir_2":20,"nn_MSeg_phiDir_2":20,"nn_MSeg_t0_2":200,"nn_MSeg_etaPos_3":20,"nn_MSeg_phiPos_3":20,"nn_MSeg_etaDir_3":20,"nn_MSeg_phiDir_3":20,"nn_MSeg_t0_3":200,"nn_MSeg_etaPos_4":20,"nn_MSeg_phiPos_4":20,"nn_MSeg_etaDir_4":20,"nn_MSeg_phiDir_4":20,"nn_MSeg_t0_4":200,"l1_ecal_0":20,"l2_ecal_0":20,"l3_ecal_0":20,"l4_ecal_0":20,"l1_hcal_0":20,"l2_hcal_0":20,"l3_hcal_0":20,"l4_hcal_0":20,"l1_ecal_1":20,"l2_ecal_1":20,"l3_ecal_1":20,"l4_ecal_1":20,"l1_hcal_1":20,"l2_hcal_1":20,"l3_hcal_1":20,"l4_hcal_1":20,"l1_ecal_2":20,"l2_ecal_2":20,"l3_ecal_2":20,"l4_ecal_2":20,"l1_hcal_2":20,"l2_hcal_2":20,"l3_hcal_2":20,"l4_hcal_2":20,"l1_ecal_2":20,"l2_ecal_3":20,"l3_ecal_3":20,"l4_ecal_3":20,"l1_hcal_3":20,"l2_hcal_3":20,"l3_hcal_3":20,"l4_hcal_3":20,"l1_ecal_4":20,"l2_ecal_4":20,"l3_ecal_4":20,"l4_ecal_4":20,"l1_hcal_4":20,"l2_hcal_4":20,"l3_hcal_4":20,"l4_hcal_4":20}

    truth_bin_dict = {"aux_llp_Lxy":20,"aux_llp_Lz":20,"aux_llp_pt":20,"aux_llp_eta":20,"aux_llp_phi":20}

    print("Plotting Variable Plots")
    print(signal)

    for key in truth_bin_dict:
        pass
       #do_plotting_truth(signal, key, truth_bin_dict[key], signal_right,model_to_do)

    for key in xmin_dict:
        pass
       #do_plotting(signal,qcd,bib,key,xmin_dict[key],xmax_dict[key],bin_dict[key], signal_right, qcd_right, bib_right, model_to_do)


 
    '''
    filter_nn_clus = [col for col in signal if col.startswith("clus_pt")]
    filter_nn_track = [col for col in signal if col.startswith("nn_track_pt")]
    filter_nn_MSeg = [col for col in signal if col.startswith("nn_MSeg_etaDir")]

    plot_three_histos( ( (signal[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_clus].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_constits", 0, 30, 30, prefix)
    plot_three_histos( (  (signal[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_track].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_tracks", 0, 20, 20, prefix)
    plot_three_histos( ( (signal[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (qcd[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), ( (bib[filter_nn_MSeg].fillna(0)).astype(bool).sum(axis=1)).values.flatten(), "n_MuonSegments", 0, 70, 70, prefix)
    '''
    





