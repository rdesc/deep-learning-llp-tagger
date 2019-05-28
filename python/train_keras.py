import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')

import tensorflow

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Highway, Dropout, Masking, LSTM, Convolution1D, Convolution2D, Flatten, Input, Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.regularizers import l1, l2
from keras.utils import np_utils

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


y_test = np_utils.to_categorical(Y)

model = Sequential()

model.add(Dense(300, input_dim=X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dense(102))
model.add(Activation('relu'))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dense(6))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train.values, y_train, sample_weight= weights_train.values, epochs=50, batch_size=512, validation_data = (X_test.values, y_test, weights_test.values),callbacks=[
                        EarlyStopping(
                            verbose=True,
                            patience=5,
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

evaluate_model(X_test, y_test, weights_test)

