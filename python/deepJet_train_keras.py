import os

import matplotlib

matplotlib.use('agg')

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, concatenate
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from utils import load_dataset

os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'
os.environ['exception_verbosity'] = 'high'


def keras_setup():
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=64)
    tf.set_random_seed(1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)


def train_llp(filename, useGPU2, constit_input, track_input, MSeg_input, jet_input,
              frac=1.0, reg_value=0.001, dropout_value=0.1, epochs=50, learning_rate=0.002, hidden_fraction=1):
    # TODO: Delete time?
    # TODO: with parametrization?

    # Do Keras_setup
    print("Setting up Keras...\n")
    keras_setup()

    # Choose GPU
    if useGPU2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Load dataset
    print("\nLoading up dataset " + filename + "...\n")
    df = load_dataset(filename)

    # Extract labels
    Y = df['label']
    # Use pt flattened weights from pre-processing for weights
    weights = df['flatWeight']  # TODO: what are these weights for?
    # Keep mcWeights TODO: what is this? for evaluation
    mcWeights = df['mcEventWeight']
    # Hard code start and end of names of variables
    X = df.loc[:, 'clus_pt_0':'nn_MSeg_t0_29']
    X = df.loc[:, 'jet_pt':'jet_phi'].join(X)

    # Label Z as parametrized variables
    Z = df.loc[:, 'llp_mH':'llp_mS']
    mass_array = (df.groupby(['llp_mH', 'llp_mS']).size().reset_index().rename(columns={0: 'count'}))
    mass_array['proportion'] = mass_array['count'] / len(df.index)

    # Save memory
    del df

    # Split data into train/test datasets
    X_train, X_test, y_train, y_test, weights_train, weights_test, mcWeights_train, mcWeights_test, Z_train, Z_test = \
        train_test_split(X, Y, weights, mcWeights, Z, test_size=0.2)

    # Keep fraction of events specified by frac param
    X_train = X_train.iloc[0:int(X_train.shape[0] * frac)]
    y_train = y_train.iloc[0:int(y_train.shape[0] * frac)]
    weights_train = weights_train.iloc[0:int(weights_train.shape[0] * frac)]
    mcWeights_train = mcWeights_train.iloc[0:int(mcWeights_train.shape[0] * frac)]
    Z_train = Z_train.iloc[0:int(Z_train.shape[0] * frac)]

    # Divide testing set into epoch-by-epoch validation and final evaluation sets
    X_test, X_val, y_test, y_val, weights_test, weights_val, mcWeights_test, mcWeights_val, Z_test, Z_val = \
        train_test_split(X_test, y_test, weights_test, mcWeights_test, Z_test, test_size=0.5)

    # Delete variables we don't need anymore (need to save memory...)
    del X
    del Y
    del Z

    # Convert labels to categorical (needed for multiclass training)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    # Split X into track, MSeg, and constit inputs and reshape dataframes into shape expected by Keras
    # This is an ordered array, so each input is formatted as number of constituents x number of variables
    print("\nPreparing train, test, and validate data for model...\n")
    print("\nPreparing constit data...")
    X_train_constit, X_test_constit, X_val_constit = constit_input.extract_and_split_data(X_train, X_test, X_val,
                                                                                          'clus_pt_0', 'clus_time_')
    print("\nPreparing track data...")
    X_train_track, X_test_track, X_val_track = track_input.extract_and_split_data(X_train, X_test, X_val,
                                                                                  'nn_track_pt_0', 'nn_track_SCTHits_')
    print("\nPreparing MSeg data...")
    X_train_MSeg, X_test_MSeg, X_val_MSeg = MSeg_input.extract_and_split_data(X_train, X_test, X_val,
                                                                              'nn_MSeg_etaPos_0', 'nn_MSeg_t0_')
    print("\nPreparing jet data...")
    X_train_jet, X_test_jet, X_val_jet = jet_input.extract_and_split_data(X_train, X_test, X_val, 'jet_pt', 'jet_phi')

    # Done preparing inputs for model!!
    print("\nDone preparing data for model!!!\n")

    # Now to setup ML architecture
    print("\nSetting up model architecture...\n")
    # input shape = 3D tensor with shape: (batch, steps, channels)
    # output shape = 3D tensor with shape: (batch, new_steps, filters

    # Set up inputs and outputs for Conv1D layers
    constit_input_tensor, constit_output_tensor = constit_input.init_keras_cnn_input_output(X_train_constit[0].shape)
    track_input_tensor, track_output_tensor = track_input.init_keras_cnn_input_output(X_train_track[0].shape)
    MSeg_input_tensor, MSeg_ouput_tensor = MSeg_input.init_keras_cnn_input_output(X_train_MSeg[0].shape)

    # Set up LSTM layers + Dense layer for monitoring
    constit_output_tensor = constit_input.init_keras_lstm(reg_value, constit_output_tensor)
    track_output_tensor = track_input.init_keras_lstm(reg_value, track_output_tensor)
    MSeg_ouput_tensor = MSeg_input.init_keras_lstm(reg_value, MSeg_ouput_tensor)

    # Set up layers for jet
    jet_input_tensor, jet_output_tensor = jet_input.init_keras_dense_input_output(X_train_jet.values[0].shape)

    # Setup concatenation layer
    concat_tensor = concatenate([constit_output_tensor, track_output_tensor, MSeg_ouput_tensor, jet_output_tensor])

    # Setup Dense + Dropout layers
    concat_tensor = Dense(hidden_fraction * 512, activation='relu')(concat_tensor)
    concat_tensor = Dropout(dropout_value)(concat_tensor)
    concat_tensor = Dense(hidden_fraction * 64, activation='relu')(concat_tensor)
    concat_tensor = Dropout(dropout_value)(concat_tensor)

    # Setup final layer
    main_output_tensor = Dense(3, activation='softmax', name='main_output')(concat_tensor)

    # Setup training
    layers_to_input = [constit_input_tensor, track_input_tensor, MSeg_input_tensor, jet_input_tensor]
    layers_to_output = [main_output_tensor, constit_output_tensor, track_output_tensor, MSeg_ouput_tensor,
                        jet_output_tensor]
    x_to_train = [X_train_constit, X_train_track, X_train_MSeg, X_train_jet.values]
    y_to_train = [y_train, y_train, y_train, y_train, y_train]
    weights_to_train = [weights_train.values, weights_train.values, weights_train.values, weights_train.values,
                        weights_train.values]

    # Setup validation
    x_to_validate = [X_test_constit, X_test_track, X_test_MSeg, X_test_jet.values]
    y_to_validate = [y_test, y_test, y_test, y_test, y_test]
    weights_to_validate = [weights_test.values, weights_test.values, weights_test.values, weights_test.values,
                           weights_test.values]
    weights_for_loss = [1., 0.01, 0.4, 0.1, 0.01]

    # Setup Model
    model = Model(inputs=layers_to_input, outputs=layers_to_output)

    # Setup optimiser (Nadam is good as it has decaying learning rate) TODO: to look into maybe
    optimizer = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    # Compile Model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', loss_weights=weights, metrics=['accuracy'])

    # Show summary of model architecture
    print(model.summary())
