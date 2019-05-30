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


session_conf = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=64)
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
keras.backend.set_session(sess)

df = pd.read_pickle("processed_output")
df = df.fillna(0)
df = df.sample(frac=1).reset_index(drop=True)

del df['track_sign']
del df['sum_eFrac']
del df['clus_sign']

print("Length of Signal is: " + str(df[df.label==1].shape[0]) )
print("Length of QCD is: " + str(df[df.label==0].shape[0]) )
print("Length of BIB is: " + str(df[df.label==2].shape[0]) )

Y = df['label']
weights = df['flatWeight']
X= df.iloc[:,3:df.shape[1]]

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, Y, weights, test_size = 0.1)

model_to_do = "lstm"


y_val = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)

if(model_to_do == "lstm"):

    X_train_constit = X_train.loc[:,'clus_pt_0':'clusTime_29']
    X_train_track = X_train.loc[:,'nn_track_pt_0':'nn_track_SCTHits_19']
    X_train_MSeg = X_train.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_69']
    X_train_jet = X_train.loc[:,'jet_pt':'jet_phi']

    X_test_constit = X_test.loc[:,'clus_pt_0':'clusTime_29']
    X_test_track = X_test.loc[:,'nn_track_pt_0':'nn_track_SCTHits_19']
    X_test_MSeg = X_test.loc[:,'nn_MSeg_etaPos_0':'nn_MSeg_t0_69']
    X_test_jet = X_test.loc[:,'jet_pt':'jet_phi']

    num_constit_vars = 28
    num_track_vars = 12
    num_MSeg_vars = 5

    num_max_constits = 30
    num_max_tracks = 20
    num_max_MSegs = 70

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

    jet_input = Input(shape = X_train_jet.values[0].shape, name='jet_input')

    x = keras.layers.concatenate([constit_out, track_out, MSeg_out, jet_input])

    x = Dense(120, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(12, activation='relu')(x)
    x = Dropout(0.2)(x)

    main_output = Dense(3, activation='softmax', name='main_output')(x)

    model = Model(inputs=[constit_input, track_input, MSeg_input, jet_input], outputs=[main_output, constit_output, track_output, MSeg_output])

    plot_model(model, to_file='plots/model_plot.png', show_shapes=True, show_layer_names=True)

    n_optimizer = keras.optimizers.Nadam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.0004)
    model.compile(optimizer=n_optimizer, loss='categorical_crossentropy',
        loss_weights=[1., 0.2, 0.2, 0.2], metrics=['accuracy'])
    model.summary()

    history = model.fit([X_train_constit, X_train_track, X_train_MSeg, X_train_jet.values], [y_train, y_train, y_train, y_train], sample_weight= [weights_train.values, weights_train.values, weights_train.values, weights_train.values], epochs=20, batch_size=512, validation_data = ([X_test_constit, X_test_track, X_test_MSeg, X_test_jet.values], [y_val, y_val, y_val, y_val], [weights_test.values, weights_test.values, weights_test.values,weights_test.values]),callbacks=[
                        EarlyStopping(
                            verbose=True,
                            patience=5,
                            monitor='val_main_output_acc'),
                        ModelCheckpoint(
                            'keras_outputs/checkpoint',
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
    plt.savefig("plots/accuracy_monitoring"+ ".pdf", format='pdf', transparent=True)
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
    plt.savefig("plots/loss_monitoring"+ ".pdf", format='pdf', transparent=True)

    evaluate_model(X_test, y_test, weights_test, model_to_do)

    

if (model_to_do == "dense"):

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
                            'keras_outputs/checkpoint',
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

    evaluate_model(X_test, y_test, weights_test, model_to_do)
