import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')

import tensorflow as tf
from keras.backend import tensorflow_backend as K

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Highway, Dropout, Masking, LSTM, Convolution1D, Convolution2D, Flatten, Input, Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.regularizers import l1, l2
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

np.set_printoptions(threshold='nan')

session_conf = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=64)
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

df = pd.read_pickle("/data/fcormier/calRatio/fullRun2/transformOutput/layerChanges/processed_output.pkl")
df = df.fillna(0)

 

del df['track_sign']
del df['clus_sign']

auxDelete = [col for col in df if col.startswith("aux")]
for item in auxDelete:
    del df[item]

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
X= df.loc[:,'jet_pt':'l4_hcal_29']
del X['llp_mS']
del X['llp_mH']
del X['jet_isClean_LooseBadLLP']
Z = df.loc[:,'llp_mH':'llp_mS']


X_train, X_test, y_train, y_test, weights_train, weights_test, Z_train, Z_test = train_test_split(X, Y, weights, Z, test_size = 0.1)

del X
del Y
del Z

#model_to_do = "lstm_layerChanges_"+str(deleteTime)+"DeleteTime"
model_to_do = "dense"


y_val = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)

if("lstm" in model_to_do):

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

    num_max_constits = 30
    num_max_tracks = 20
    num_max_MSegs = 70

    if deleteTime:
        X_train_constit = X_train.loc[:,'clus_pt_0':'clus_phi_29']
    else:
        X_train_constit = X_train.loc[:,'clus_pt_0':'clusTime_29']

    X_train_track = X_train.loc[:,'nn_track_pt_0':'nn_track_SCTHits_19']
    if deleteTime:
        X_train_MSeg = X_train.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_phiDir_69']
    else:
        X_train_MSeg = X_train.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_69']
    X_train_jet = X_train.loc[:,'jet_pt':'jet_phi']

    #X_train_jet.join(X_train.loc[:,'llp_mH'])
    #X_train_jet.join(X_train.loc[:,'llp_mS'])

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

    for i in range(0,num_max_constits):
        temp_loc = X_train_constit.columns.get_loc('clus_phi_'+str(i))

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
    #X_test_jet.join(X_test.loc[:,'llp_mH'])
    #X_test_jet.join(X_test.loc[:,'llp_mS'])


    X_train_constit = X_train_constit.values.reshape(X_train_constit.shape[0],num_max_constits,num_constit_vars)
    X_train_track = X_train_track.values.reshape(X_train_track.shape[0],num_max_tracks,num_track_vars)
    X_train_MSeg = X_train_MSeg.values.reshape(X_train_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)

    X_test_constit = X_test_constit.values.reshape(X_test_constit.shape[0],num_max_constits,num_constit_vars)
    X_test_track = X_test_track.values.reshape(X_test_track.shape[0],num_max_tracks,num_track_vars)
    X_test_MSeg = X_test_MSeg.values.reshape(X_test_MSeg.shape[0],num_max_MSegs,num_MSeg_vars)


    constit_input = Input(shape=(X_train_constit[0].shape),dtype='float32',name='constit_input')
    constit_out = LSTM(num_constit_vars)(constit_input)
    constit_output = Dense(3, activation='softmax', name='constit_output')(constit_out)

    track_input = Input(shape=(X_train_track[0].shape),dtype='float32',name='track_input')
    track_out = LSTM(num_track_vars)(track_input)
    track_output = Dense(3, activation='softmax', name='track_output')(track_out)

    MSeg_input = Input(shape=(X_train_MSeg[0].shape),dtype='float32',name='MSeg_input')
    MSeg_out = LSTM(num_MSeg_vars)(MSeg_input)
    MSeg_output = Dense(3, activation='softmax', name='MSeg_output')(MSeg_out)

    print(X_train_jet)

    jet_input = Input(shape = X_train_jet.values[0].shape, name='jet_input')
    jet_out = Dense(3)(jet_input)
    jet_output = Dense(3, activation='softmax', name='jet_output')(jet_out)

    x = keras.layers.concatenate([constit_out, track_out, MSeg_out, jet_input])

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    main_output = Dense(3, activation='softmax', name='main_output')(x)

    model = Model(inputs=[constit_input, track_input, MSeg_input, jet_input], outputs=[main_output, constit_output, track_output, MSeg_output, jet_output])

    #plot_model(model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer='Adadelta', loss='categorical_crossentropy',
        loss_weights=[1., 0.1, 0.4, 0.2,0.1], metrics=['accuracy'])
    model.summary()
    '''
    history = model.fit([X_train_constit, X_train_track, X_train_MSeg, X_train_jet.values], [y_train, y_train, y_train, y_train, y_train], sample_weight= [weights_train.values, weights_train.values, weights_train.values, weights_train.values, weights_train.values], epochs=20, batch_size=512, validation_data = ([X_test_constit, X_test_track, X_test_MSeg, X_test_jet.values], [y_val, y_val, y_val, y_val,y_val], [weights_test.values, weights_test.values, weights_test.values,weights_test.values,weights_test.values]),callbacks=[
                        EarlyStopping(
                            verbose=True,
                            patience=5,
                            monitor='val_main_output_acc'),
                        ModelCheckpoint(
                            'keras_outputs/checkpoint_'+model_to_do,
                            monitor='val_main_output_acc',
                            verbose=True,
                            save_best_only=True)])
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(history.history['main_output_acc'])
    plt.plot(history.history['val_main_output_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("plots/accuracy_monitoring"+ model_to_do +".pdf", format='pdf', transparent=True)
    plt.clf()
    plt.cla()
    plt.figure()
    # summarize history for loss
    plt.plot(history.history['main_output_loss'])
    plt.plot(history.history['val_main_output_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("plots/loss_monitoring"+ model_to_do +".pdf", format='pdf', transparent=True)

    '''
    evaluate_model(X_test, y_test, weights_test, Z_test, model_to_do, deleteTime)

    

if (model_to_do == "dense"):

    print(list(X_train.columns))

    model = Sequential()

    model.add(Dense(600, input_dim=X_train.shape[1]))
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
    history = model.fit(X_train.values, y_train, sample_weight= weights_train.values, epochs=100, batch_size=512, validation_data = (X_test.values, y_val, weights_test.values),callbacks=[
                        EarlyStopping(
                            verbose=True,
                            patience=20,
                            monitor='val_acc'),
                        ModelCheckpoint(
                            'keras_outputs/checkpoint_'+model_to_do,
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
    plt.savefig("plots/accuracy_monitoring"+ ".pdf", format='pdf', transparent=True)
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
    plt.savefig("plots/loss_monitoring"+ ".pdf", format='pdf', transparent=True)
    evaluate_model(X_test, y_test, weights_test, Z_test,  model_to_do, deleteTime)
