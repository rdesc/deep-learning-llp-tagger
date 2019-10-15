import os
import numpy as np
import math
import pdb
import itertools

from random import shuffle
from pprint import pprint


import matplotlib
import matplotlib as mpl
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter, ScalarFormatter


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Highway, Dropout, Masking, LSTM, SimpleRNN, Input, CuDNNLSTM
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.regularizers import l1, l2, L1L2

import sklearn
from sklearn.metrics import roc_curve, auc

#from top_ml import plot_histo
from random import shuffle, seed
from keras.utils import np_utils
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.mplot3d import Axes3D



import sys
import seaborn as sns
import re
import sklearn

from sklearn.preprocessing import label_binarize

from numpy import unravel_index

def find_threshold(prediction,y, weight, perc, label):


    #Instead of lame loops let's order our data, then find percentage from there
    #prediction is 3xN, want to sort by BIB weight

    label_events_y = y[y==label]
    label_events_prediction = prediction[y==label]

    prediction_sorted = np.array(label_events_prediction[label_events_prediction[:,label].argsort()])
    y_sorted = label_events_y[label_events_prediction[:,label].argsort()]

    #print(prediction)
    #print(prediction[:,2])
    #print(prediction[prediction[:,2].argsort()])

    cutoffIndex = round(((100-perc)/100)*label_events_y.size)
    print("CutoffIndex: " + str(int(cutoffIndex)))
    threshold = prediction_sorted.item((int(cutoffIndex),label))
    print("Treshold: " + str(threshold))

    leftovers = np.where(
                    np.greater(
                        threshold,
                        prediction[:,label]))


    return threshold, leftovers

def signal_llp_efficiencies(prediction,y_test,Z_test,destination,f):
    sig_rows = np.where(y_test==1)
    prediction = prediction[sig_rows]
    Z_test = Z_test.iloc[sig_rows]
    mass_array = (Z_test.groupby(['llp_mH','llp_mS']).size().reset_index().rename(columns={0:'count'}))

    plot_x = []
    plot_y = []
    plot_z = []

    for item,mH,mS in zip(mass_array['count'],mass_array['llp_mH'],mass_array['llp_mS']):
        temp_array = prediction[ (Z_test['llp_mH'] == mH) & (Z_test['llp_mS'] == mS) ]
        temp_max = np.argmax(temp_array,axis=1)
        temp_num_signal_best = len(temp_max[temp_max==1])
        temp_eff = temp_num_signal_best / temp_array.shape[0]
        plot_x.append(mH)
        plot_y.append(temp_eff)
        plot_z.append(mS)
        print("mH: " + str(mH) + ", mS: " + str(mS) + ", Eff: " + str(temp_eff))
        f.write("%s,%s,%s\n" % (str(mH), str(mS), str(temp_eff)) )

    plt.clf()
    plt.figure()
    plt.scatter(plot_x, plot_y, marker='+', s=150, linewidths=4, c=plot_z, cmap=plt.cm.coolwarm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'mS')
    plt.xlabel("mH")
    plt.ylabel("Signal Efficiency")

    plt.savefig(destination+"signal_llp_efficiencies"+ ".pdf", format='pdf', transparent=True)
    plt.clf()


def make_multi_roc_curve(prediction,y,weight,threshold,label,leftovers):
    tag_eff = []
    bkg_eff = []

    sig_tot = np.sum(np.where(np.equal(y, 1), weight, 0))
    qcd_tot = np.sum(np.where(np.equal(y, 0), weight, 0))
    bib_tot = np.sum(np.where(np.equal(y, 2), weight, 0))
    prediction_left = prediction[leftovers]
    y_left = y.values[leftovers]
    print(len(y_left))
    weight_left = weight.values[leftovers]

    num_signal_original = y[y==1].size
    num_signal_leftover = y_left[y_left==1].size
    signal_ratio = num_signal_leftover/num_signal_original

    num_qcd_original = np.sum(weight.values[y==0])
    num_qcd_leftover = np.sum(weight_left[y_left==0])
    qcd_ratio = num_qcd_leftover/num_qcd_original

    num_bib_original = np.sum(weight.values[y==2])
    num_bib_leftover = np.sum(weight_left[y_left==2])
    bib_ratio = num_bib_leftover/num_bib_original


    sig_left = np.sum(np.where(np.equal(y_left, 1), weight_left, 0))
    qcd_left = np.sum(np.where(np.equal(y_left, 0), weight_left, 0))
    bib_left = np.sum(np.where(np.equal(y_left, 2), weight_left, 0))
    prediction_left_signal = prediction_left[:,1]
    prediction_left_qcd = prediction_left[:,0]
    prediction_left_bib = prediction_left[:,2]
    #y_left[y_left==2] = 0

    if label == 2:
        
        y_roc = label_binarize(y_left, classes=[0, 1, 2])
        (fpr, tpr, _) = roc_curve(y_roc[:,1], prediction_left_signal,  pos_label=1)
        #(fpr, tpr, _) = roc_curve(y_left, prediction_left_signal, sample_weight=weight_left, pos_label=1)
        #print(str(list(1-fpr[1000:1100])))
        #print(str(list(tpr[1000:1100])))
        a = auc((1-fpr)*qcd_ratio,tpr*signal_ratio)
        
        return (1/fpr)*qcd_ratio, tpr*signal_ratio, a


    if label == 0:
        
        y_roc = label_binarize(y_left, classes=[0, 1, 2])
        (fpr, tpr, _) = roc_curve(y_roc[:,1], prediction_left_signal, sample_weight=weight_left, pos_label=1)
        #print(str(list(1-fpr[1000:1100])))
        #print(str(list(tpr[1000:1100])))
        a = auc((1-fpr)*bib_ratio,tpr*signal_ratio)
        
        return (1/fpr)*bib_ratio, tpr*signal_ratio, a


def get_efficiencies_with_weights(py, y ,weight, threshold):
            y.astype(int)
            #print("py examples:")
            #print(len(py))
            #print(py[0:10])
            #print("y examples:")
            #print(len(y))
            #print(y[0:10])

            S = np.sum(np.where(np.equal(y, 1.0), weight, 0))
            B = np.sum(np.where(np.equal(y, 0.0), weight, 0))
            accuracy = np.sum(np.where(np.not_equal(
                y, np.greater(py, y)), 0, weight)) / float(len(py))
            s = np.sum(
                np.where(
                    np.logical_and(
                        y == 1.0,
                        np.greater(
                            py,
                            threshold)),
                    weight,
                    0))
            b = np.sum(
                np.where(
                    np.logical_and(
                        y == 0.0,
                        np.greater(
                            py,
                            threshold)),
                    weight,
                    0))
            sig_eff = s / S
            sig_err = math.sqrt((s * (1 - s / S))) / S
            #bg_eff = b / B
            #bg_err = math.sqrt((b * (1 - b / B))) / B
            bg_rej = B / b
            bg_rej_err = get_reg_errorbar(b,B)
            #print("s = " + str(s))
            #print("S = " + str(S))
            #print("b = " + str(b))
            #print("B = " + str(B))
            #print("Accuracy: " + str(accuracy))
            return sig_eff, bg_rej, sig_err, bg_rej_err, S, B


def plot_prediction_histograms(destination,
                                prediction,
                                labels, weight, model_to_do):

    sig_rows = np.where(labels==1)
    bkg_rows = np.where(labels==0)
    bib_rows = np.where(labels==2)
    #print(sig_rows)
    plt.clf()

    fig,ax = plt.subplots()
    textstr = model_to_do 

    #ax.hist(prediction[sig_rows][:,1], weights=weight.values[sig_rows], color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=50,label="Signal")
    #ax.hist(prediction[bkg_rows][:,1], weights=weight.values[bkg_rows], color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=50,label="QCD")
    #ax.hist(prediction[bib_rows][:,1], weights=weight.values[bib_rows], color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=50,label="BIB")
    bin_list = np.zeros(1)
    bin_list = np.append(bin_list,np.logspace(np.log10(0.00001),np.log10(1.0), 50))
    ax.hist(prediction[sig_rows][:,1], color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=bin_list,label="Signal")
    ax.hist(prediction[bkg_rows][:,1], color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bin_list,label="QCD")
    ax.hist(prediction[bib_rows][:,1], color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bin_list,label="BIB")
    plt.yscale('log', nonposy='clip')
    plt.xscale('log', nonposx='clip')
    ax.set_xlabel("Signal NN weight")
    ax.legend(loc='best')

    #matplotlib.patch.Patch properties
    '''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    '''
    # place a text box in upper left in axes coords
    '''
    ax.text(0.05, 0.8, textstr, color='black', transform=ax.transAxes, 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    '''

    plt.savefig(destination+"sig_predictions"+ ".pdf", format='pdf', transparent=True)
    plt.clf()


    fig,ax = plt.subplots()
    ax.hist(prediction[sig_rows][:,0], weights=weight.values[sig_rows], color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=bin_list,label="Signal")
    ax.hist(prediction[bkg_rows][:,0], weights=weight.values[bkg_rows], color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bin_list,label="QCD")
    ax.hist(prediction[bib_rows][:,0], weights=weight.values[bib_rows], color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bin_list,label="BIB")
    plt.yscale('log', nonposy='clip')
    ax.set_xlabel("QCD NN weight")
    plt.xscale('log', nonposx='clip')
    ax.legend()

    #matplotlib.patch.Patch properties
    '''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    '''
    # place a text box in upper left in axes coords
    '''
    ax.text(0.05, 0.8, textstr, color='black', transform=ax.transAxes, 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    '''

    plt.savefig(destination+"qcd_predictions"+ ".pdf", format='pdf', transparent=True)
    plt.clf()


    fig,ax = plt.subplots()
    ax.hist(prediction[sig_rows][:,2], weights=weight.values[sig_rows], color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=bin_list,label="Signal")
    ax.hist(prediction[bkg_rows][:,2], weights=weight.values[bkg_rows], color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bin_list,label="QCD")
    ax.hist(prediction[bib_rows][:,2], weights=weight.values[bib_rows], color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=bin_list,label="BIB")
    plt.yscale('log', nonposy='clip')
    ax.set_xlabel("BIB NN weight")
    plt.xscale('log', nonposx='clip')
    ax.legend()

    #matplotlib.patch.Patch properties
    '''
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.8, textstr, color='black', transform=ax.transAxes, 
        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    '''

    plt.savefig(destination+"bib_predictions"+ ".pdf", format='pdf', transparent=True)
    plt.clf()
    return

def evaluate_model(X_test, y_test, weights_test, mcWeights_test,  Z_test,  model_to_do, deleteTime, num_constit_lstm, num_track_lstm, num_mseg_lstm, reg_value, doTrackLSTM, doMSegLSTM, doParametrization, learning_rate):

    if ("dense" in model_to_do):

        model = Sequential()

        model.add(Dense(2000, input_dim=X_test.shape[1]))
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
        model.load_weights("keras_outputs/"+model_to_do+'/checkpoint')

    if ("lstm" in model_to_do):

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
            X_test_constit = X_test.loc[:,'clus_pt_0':'clus_phi_29']
        else:
            X_test_constit = X_test.loc[:,'clus_pt_0':'clusTime_29']
        X_test_track = X_test.loc[:,'nn_track_pt_0':'nn_track_SCTHits_19']
        if deleteTime:
            X_test_MSeg = X_test.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_69']
        else:
            X_test_MSeg = X_test.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_69']
        X_test_jet = X_test.loc[:,'jet_pt':'jet_phi']
        if doParametrization:
            X_test_jet = X_test_jet.join(Z_test)

        for i in range(0,num_max_constits):
            temp_loc = X_test_constit.columns.get_loc('clus_phi_'+str(i))

            X_test_constit.insert(temp_loc,'l4_hcal_'+str(i),X_test['l4_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l3_hcal_'+str(i),X_test['l3_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l2_hcal_'+str(i),X_test['l2_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l1_hcal_'+str(i),X_test['l1_hcal_'+str(i)])
            X_test_constit.insert(temp_loc,'l4_ecal_'+str(i),X_test['l4_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l3_ecal_'+str(i),X_test['l3_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l2_ecal_'+str(i),X_test['l2_ecal_'+str(i)])
            X_test_constit.insert(temp_loc,'l1_ecal_'+str(i),X_test['l1_ecal_'+str(i)])


        X_test_constit = X_test_constit.values.reshape(X_test_constit.shape[0],num_max_constits,num_constit_vars)
        X_test_track = X_test_track.values.reshape(X_test_track.shape[0],num_max_tracks,num_track_vars)
        X_test_MSeg = X_test_MSeg.values.reshape(X_test_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)


        print( "CONSTIT SIZE: " + str(X_test_constit[0].shape) )

        constit_input = Input(shape=(X_test_constit[0].shape),dtype='float32',name='constit_input')
        constit_out = LSTM(num_constit_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(constit_input)
        constit_output = Dense(3, activation='softmax', name='constit_output')(constit_out)

        track_input = Input(shape=(X_test_track[0].shape),dtype='float32',name='track_input')
        track_out = LSTM(num_track_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(track_input)
        track_output = Dense(3, activation='softmax', name='track_output')(track_out)

        MSeg_input = Input(shape=(X_test_MSeg[0].shape),dtype='float32',name='MSeg_input')
        MSeg_out = LSTM(num_mseg_lstm, kernel_regularizer = L1L2(l1=reg_value, l2=reg_value))(MSeg_input)
        MSeg_output = Dense(3, activation='softmax', name='MSeg_output')(MSeg_out)

        jet_input = Input(shape = X_test_jet.values[0].shape, name='jet_input')
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
        x_to_validate = [X_test_constit, X_test_track, X_test_MSeg, X_test_jet.values]
        y_to_validate = [y_test, y_test, y_test, y_test,y_test]
        weights_to_validate = [weights_test.values, weights_test.values, weights_test.values,weights_test.values,weights_test.values]
        weights_for_loss = [1., 0.1, 0.4, 0.2,0.1]

        if (doTrackLSTM and not doMSegLSTM):
            layers_to_input = [constit_input, track_input,  jet_input]
            layers_to_output = [main_output, constit_output, track_output, jet_output]
            x_to_validate = [X_test_constit, X_test_track, X_test_jet.values]
            y_to_validate = [y_test, y_test, y_test, y_test]
            weights_to_validate = [weights_test.values,  weights_test.values,weights_test.values,weights_test.values]
            weights_for_loss = [1., 0.1, 0.4, 0.1]
        if (doMSegLSTM and not doTrackLSTM):
            print("HERE")
            layers_to_input = [constit_input, MSeg_input,  jet_input]
            layers_to_output = [main_output, constit_output, MSeg_output, jet_output]
            x_to_validate = [X_test_constit, X_test_MSeg, X_test_jet.values]
            y_to_validate = [y_test, y_test, y_test, y_test]
            weights_to_validate = [weights_test.values,  weights_test.values,weights_test.values,weights_test.values]
            weights_for_loss = [1., 0.1, 0.2,0.1]
        if (not doTrackLSTM and not doMSegLSTM):
            layers_to_input = [constit_input,  jet_input]
            layers_to_output = [main_output, constit_output, jet_output]
            x_to_validate = [X_test_constit, X_test_jet.values]
            y_to_validate = [y_test, y_test,  y_test]
            weights_to_validate = [weights_test.values,weights_test.values,weights_test.values]
            weights_for_loss = [1., 0.1, 0.1]

        model = Model(inputs=layers_to_input, outputs=layers_to_output)


        opt = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(optimizer=opt, loss='categorical_crossentropy',
            loss_weights=weights_for_loss, metrics=['accuracy'])
        model.load_weights("keras_outputs/"+model_to_do+'/checkpoint')

    if ("dense" in model_to_do):
	    prediction = model.predict(X_test.values, verbose=True)

    elif ("lstm" in model_to_do):
        prediction = model.predict(x_to_validate, verbose = True)
        prediction = prediction[0]

    #print(y_test==2)
    #print(weights_test)

    bib_weight = np.sum(mcWeights_test[y_test==2])
    sig_weight = np.sum(mcWeights_test[y_test==1])
    qcd_weight = np.sum(mcWeights_test[y_test==0])

    bib_weight_length = len(mcWeights_test[y_test==2])
    sig_weight_length = len(mcWeights_test[y_test==1])
    qcd_weight_length = len(mcWeights_test[y_test==0])

    mcWeights_test[y_test==0] *= qcd_weight_length/qcd_weight
    mcWeights_test[y_test==2] *= bib_weight_length/bib_weight
    mcWeights_test[y_test==1] *= sig_weight_length/sig_weight
    destination = "plots/"+model_to_do + "/"
    plot_prediction_histograms(destination,prediction,y_test, mcWeights_test, model_to_do)
    #threshold_array = np.logspace(-0.1,-0.001,30)[::-3]
    threshold_array = [0.9999,(1-0.001),(1-0.003),(1-0.009),(1-0.023),(1-0.059),(1-0.151),(1-0.389),0.001]
    #threshold_array = [0.995,(1-0.009),(1-0.023),(1-0.059),(1-0.151),(1-0.389),0.001]
    counter=0
    third_label=2
    f = open(destination+"training_details.txt","a")


    #threshold_array = [0.99,(1-0.03),(1-0.09),(1-0.23),(1-0.59),0.001]
    #threshold_array =  np.logspace(-4,0,30)[::-3]
    #for percent in range(0,100,10):
    for item in threshold_array:
        test_perc = item*100
        test_label = third_label
        #print(prediction.shape)
        #print(y_predictions.shape)
        test_threshold, leftovers = find_threshold(prediction,y_test, mcWeights_test, test_perc, test_label)
        bkg_eff, tag_eff, roc_auc = make_multi_roc_curve(prediction,y_test,mcWeights_test,test_threshold,test_label,leftovers)
        f.write("%s, %s\n" % (str(-item+1), str(roc_auc)) )
        print("AUC: " + str(roc_auc) )
        plt.plot(tag_eff, bkg_eff, label= f"BIB Eff: {(-item+1):.3f}" +f", AUC: {roc_auc:.3f}")
        plt.xlabel("LLP Tagging Efficiency")
        axes = plt.gca()
        axes.set_xlim([0,1])
        counter=counter+1
        
        #axes.set_ylim([0,1])
    #matplotlib.patch.Patch properties
    #props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #textstr = model_to_do 
    # place a text box in upper left in axes coords
    #ax.text(0.05, 0.8, textstr, color='black', transform=ax.transAxes, 
    #    bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    signal_llp_efficiencies(prediction,y_test,Z_test, destination,f)
    plt.legend()
    plt.yscale('log', nonposy='clip')
    signal_test = prediction[y_test==1]
    qcd_test = prediction[y_test==0]
    print(signal_test[0:100].shape)
    print("Length of Signal: " + str(len(signal_test)) + ", length of signal with weight 1: " + str(len(signal_test[signal_test[:,1]<0.1])))
    print("Length of QCD: " + str(len(qcd_test)) + ", length of qcd with weight 1: " + str(len(qcd_test[qcd_test[:,1]<0.1])))
    if third_label == 2:
        plt.ylabel("QCD Rejection")
        plt.savefig(destination + "roc_curve_atlas_rej_bib" + ".pdf", format='pdf', transparent=True)
    if third_label == 0:
        plt.ylabel("BIB Rejection")
        plt.savefig(destination + "roc_curve_atlas_rej_qcd" + ".pdf", format='pdf', transparent=True)
    plt.clf()
    plt.cla()
    f.close()
    #plot_vars_final(X_test, y_test, prediction, mcWeights_test)



