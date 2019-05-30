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
from keras.layers import Dense, Activation, Highway, Dropout, Masking, LSTM, SimpleRNN, Input
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
#from keras.regularizers import l1, l2, l1l2, activity_l1l2

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

def find_threshold(prediction,y, weight, perc, label):


    #Instead of lame loops let's order our data, then find percentage from there
    #prediction is 3xN, want to sort by BIB weight

    bib_events_y = y[y==2]
    bib_events_prediction = prediction[y==2]

    prediction_sorted = np.array(bib_events_prediction[bib_events_prediction[:,2].argsort()])
    y_sorted = bib_events_y[bib_events_prediction[:,2].argsort()]

    #print(prediction)
    #print(prediction[:,2])
    #print(prediction[prediction[:,2].argsort()])

    cutoffIndex = round(((100-perc)/100)*bib_events_y.size)
    print("CutoffIndex: " + str(int(cutoffIndex)))
    threshold = prediction_sorted.item((int(cutoffIndex),2))
    print("Treshold: " + str(threshold))

    leftovers = np.where(
                    np.greater(
                        threshold,
                        prediction[:,label]))


    return threshold, leftovers

def make_multi_roc_curve(prediction,y,weight,threshold,label,leftovers):
    tag_eff = []
    bkg_eff = []

    sig_tot = np.sum(np.where(np.equal(y, 1), weight, 0))
    qcd_tot = np.sum(np.where(np.equal(y, 0), weight, 0))
    bib_tot = np.sum(np.where(np.equal(y, 2), weight, 0))
    prediction_left = prediction[leftovers]
    y_left = y.values[leftovers]
    weight_left = weight.values[leftovers]

    num_signal_original = y[y==1].size
    num_signal_leftover = y_left[y_left==1].size
    signal_ratio = num_signal_leftover/num_signal_original

    num_qcd_original = np.sum(weight.values[y==0])
    num_qcd_leftover = np.sum(weight_left[y_left==0])
    qcd_ratio = num_qcd_leftover/num_qcd_original


    sig_left = np.sum(np.where(np.equal(y_left, 1), weight_left, 0))
    qcd_left = np.sum(np.where(np.equal(y_left, 0), weight_left, 0))
    bib_left = np.sum(np.where(np.equal(y_left, 2), weight_left, 0))
    prediction_left_signal = prediction_left[:,1]
    #y_left[y_left==2] = 0
    (fpr, tpr, _) = roc_curve(y_left, prediction_left_signal, sample_weight=weight_left, pos_label=1)
    a = auc(fpr*qcd_ratio,tpr*signal_ratio)

    return fpr, tpr, a, signal_ratio, qcd_ratio


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
                                labels, weight):
    #plt.cla()
    #plt.figure()
    #print(prediction)
    #prediction = np.asarray(prediction[0])
    sig_rows = np.where(labels==1)
    bkg_rows = np.where(labels==0)
    bib_rows = np.where(labels==2)
    #print(sig_rows)
    plt.hist(prediction[sig_rows][:,1], weights=weight.values[sig_rows], color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=50,label="Signal")
    plt.hist(prediction[bkg_rows][:,1], weights=weight.values[bkg_rows], color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=50,label="QCD")
    plt.hist(prediction[bib_rows][:,1], weights=weight.values[bib_rows], color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=50,label="BIB")
    plt.xlabel("Signal NN weight")
    plt.legend()
    plt.savefig(destination+"/sig_predictions"+ ".pdf", format='pdf', transparent=True)
    plt.clf()

    plt.hist(prediction[sig_rows][:,0], weights=weight.values[sig_rows], color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=50,label="Signal")
    plt.hist(prediction[bkg_rows][:,0], weights=weight.values[bkg_rows], color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=50,label="QCD")
    plt.hist(prediction[bib_rows][:,0], weights=weight.values[bib_rows], color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=50,label="BIB")
    plt.xlabel("QCD NN weight")
    plt.legend()
    plt.savefig(destination+"/qcd_predictions"+ ".pdf", format='pdf', transparent=True)
    plt.clf()

    plt.hist(prediction[sig_rows][:,2], weights=weight.values[sig_rows], color='red',alpha=0.5,linewidth=0, histtype='stepfilled',bins=50,label="Signal")
    plt.hist(prediction[bkg_rows][:,2], weights=weight.values[bkg_rows], color='blue',alpha=0.5, linewidth=0,histtype='stepfilled',bins=50,label="QCD")
    plt.hist(prediction[bib_rows][:,2], weights=weight.values[bib_rows], color='green',alpha=0.5, linewidth=0,histtype='stepfilled',bins=50,label="BIB")
    plt.xlabel("BIB NN weight")
    plt.legend()
    plt.savefig(destination+"/bib_predictions"+ ".pdf", format='pdf', transparent=True)
    plt.clf()
    return

def evaluate_model(X_test, y_test, weights_test):

    model = Sequential()
    model.add(Dense(600, input_dim=X_test.shape[1]))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(202))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(22))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(12))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.load_weights("keras_outputs/checkpoint")

    prediction = model.predict(X_test.values, verbose=True)

    #print(y_test==2)
    #print(weights_test)

    bib_weight = np.sum(weights_test[y_test==2])
    sig_weight = np.sum(weights_test[y_test==1])
    qcd_weight = np.sum(weights_test[y_test==0])

    weights_test[y_test==0] *= sig_weight/qcd_weight
    weights_test[y_test==2] *= sig_weight/bib_weight
    destination = "plots/"
    plot_prediction_histograms(destination,prediction,y_test, weights_test)
    #threshold_array = np.logspace(-0.1,-0.001,30)[::-3]
    plt.figure()
    #threshold_array = [0.9999,(1-0.001),(1-0.003),(1-0.009),(1-0.023),(1-0.059),(1-0.151),(1-0.389),0.001]
    threshold_array = [0.99,(1-0.03),(1-0.09),(1-0.23),(1-0.59),0.001]
    #threshold_array =  np.logspace(-4,0,30)[::-3]
    #for percent in range(0,100,10):
    for item in threshold_array:
        test_perc = item*100
        test_label = 2
        #print(prediction.shape)
        #print(y_predictions.shape)
        test_threshold, leftovers = find_threshold(prediction,y_test, weights_test, test_perc, test_label)
        bkg_eff, tag_eff, roc_auc, sig_ratio, qcd_ratio = make_multi_roc_curve(prediction,y_test,weights_test,test_threshold,test_label,leftovers)
        print("AUC: " + str(roc_auc) )
        plt.plot(tag_eff*sig_ratio, (1.0-bkg_eff)*qcd_ratio, label= f"BIB Eff: {(-item+1):.3f}" +f", AUC: {roc_auc:.3f}")
        plt.xlabel("LLP Tagging Efficiency")
        plt.ylabel("QCD Efficiency")
        axes = plt.gca()
        axes.set_xlim([0,1])
        #axes.set_ylim([0,1])
    plt.legend()
    plt.savefig("plots/roc_curve_atlas_all" + ".pdf", format='pdf', transparent=True)
    plt.clf()
    plt.cla()



