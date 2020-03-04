import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, concatenate
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from utils import load_dataset, create_directories, evaluate_model

matplotlib.use('agg')

os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'
os.environ['exception_verbosity'] = 'high'


def train_llp(file_name, model_to_do, useGPU2, constit_input, track_input, MSeg_input, jet_input, plt_model=False, frac=1.0,
              batch_size=5000, reg_value=0.001, dropout_value=0.1, epochs=50, learning_rate=0.002, hidden_fraction=1, kfold=None):
    """
    Takes in arguments to change architecture of network, does training, then runs evaluate_training
    :param file_name: Name of the .pkl file containing all the data
    :param model_to_do: Name of the model
    :param useGPU2: True to use GPU2
    :param constit_input: ModelInput object for constituents
    :param track_input: ModelInput object for tracks
    :param MSeg_input: ModelInput object for muon segments
    :param jet_input: ModelInput object for jets
    :param plt_model: True to save model architecture to disk
    :param frac: Fraction of events to use in file_name
    :param batch_size: Number of training examples in one forward/backward pass
    :param reg_value: Value of regularizer term for LSTM
    :param dropout_value: Fraction of the input units to drop
    :param epochs: Number of epochs to train the model
    :param learning_rate: Learning rate
    :param hidden_fraction: Fraction by which to multiple the dense layers
    :param kfold: KFold object to do KFold cross validation
    """
    # Setup directories
    print("\nSetting up directories...\n")
    dir_name = create_directories(model_to_do, os.path.split(os.path.splitext(file_name)[0])[1])

    # Write a file with some details of architecture, will append final stats at end of training
    print("\nWriting to file training details...\n")
    f = open("plots/" + dir_name + "/training_details.txt", "w+")
    f.write("File name\n")
    f.write(file_name + "\n")
    f.write("\nModel name\n")
    f.write(model_to_do + "\n")
    f.write("\nModelInput objects\n")
    f.write(str(vars(constit_input)) + "\n")
    f.write(str(vars(track_input)) + "\n")
    f.write(str(vars(MSeg_input)) + "\n")
    f.write(str(vars(jet_input)) + "\n")
    f.write("\nOther hyperparameters\n")
    f.write("frac = %s\nbatch_size = %s\nreg_value = %s\ndropout_value = %s\nepochs = %s\nlearning_rate = %s\n"
            "hidden_fraction = %s\n" % (frac, batch_size, reg_value, dropout_value, epochs, learning_rate, hidden_fraction))
    f.close()

    # Do Keras_setup
    print("\nSetting up Keras...\n")
    keras_setup()

    # Choose GPU
    if useGPU2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Load dataset
    print("\nLoading up dataset " + file_name + "...\n")
    df = load_dataset(file_name)

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
    # mass_array = (df.groupby(['llp_mH', 'llp_mS']).size().reset_index().rename(columns={0: 'count'}))
    # mass_array['proportion'] = mass_array['count'] / len(df.index)  # TODO: never used??

    # Save memory
    del df

    # Handle case if no KFold
    if kfold is None:
        # Split data into train/test datasets
        X_train, X_test, y_train, y_test, weights_train, weights_test, mcWeights_train, mcWeights_test, Z_train, Z_test = \
            train_test_split(X, Y, weights, mcWeights, Z, test_size=0.2)

        # Delete variables to save memory
        del X
        del Y
        del Z

        # Call method that prepares data, builds model architecture, trains model, and evaluates model
        roc_auc, test_acc = build_train_evaluate_model(constit_input, track_input, MSeg_input, jet_input, X_train, X_test, y_train, y_test,
                                                       mcWeights_train, mcWeights_test, weights_train, weights_test, Z_test, Z_train, reg_value, frac,
                                                       dropout_value, hidden_fraction, plt_model, batch_size, dir_name, learning_rate, epochs)

        return roc_auc, test_acc

    else:
        # initialize lists to store metrics
        roc_scores, acc_scores = list(), list()
        # initialize counter for current fold iteration
        n_folds = 0
        # do KFold Cross Validation
        for train_ix, test_ix in kfold.split(X):
            n_folds += 1
            print("\nDoing KFold iteration # %.0f...\n" % n_folds)
            # select samples
            X_train, y_train, weights_train, mcWeights_train, Z_train = \
                X.iloc[train_ix], Y.iloc[train_ix], weights.iloc[train_ix], mcWeights.iloc[train_ix], Z.iloc[train_ix]
            X_test, y_test, weights_test, mcWeights_test, Z_test = \
                X.iloc[test_ix], Y.iloc[test_ix], weights.iloc[test_ix], mcWeights.iloc[test_ix], Z.iloc[test_ix]

            # Call method that prepares data, builds model architecture, trains model, and evaluates model
            roc_auc, test_acc = build_train_evaluate_model(constit_input, track_input, MSeg_input, jet_input, X_train, X_test, y_train, y_test,
                                                           mcWeights_train, mcWeights_test, weights_train, weights_test, Z_test, Z_train, reg_value, frac,
                                                           dropout_value, hidden_fraction, plt_model, batch_size, dir_name, learning_rate, epochs, kfold, n_folds)

            roc_scores.append(roc_auc)
            acc_scores.append(test_acc)

        return roc_scores, acc_scores


def build_train_evaluate_model(constit_input, track_input, MSeg_input, jet_input, X_train, X_test, y_train, y_test, mcWeights_train,
                               mcWeights_test, weights_train, weights_test, Z_test, Z_train, reg_value, frac, dropout_value,
                               hidden_fraction, plt_model, batch_size, dir_name, learning_rate, epochs, kfold=None, n_folds=''):
    """
    This method has the following steps:
        - Prepares train, test, and validate data
        - Builds model architecture
        - Does model training
        - Does model evaluation
    :return: ROC area under curve metric, and model accuracy metric
    """
    # Keep fraction of events specified by frac param
    X_train = X_train.iloc[0:int(X_train.shape[0] * frac)]
    y_train = y_train.iloc[0:int(y_train.shape[0] * frac)]
    weights_train = weights_train.iloc[0:int(weights_train.shape[0] * frac)]
    mcWeights_train = mcWeights_train.iloc[0:int(mcWeights_train.shape[0] * frac)]  # TODO: never used??
    Z_train = Z_train.iloc[0:int(Z_train.shape[0] * frac)]  # TODO: never used??

    if kfold is None:
        random_state = np.random.randint(100)
    else:
        random_state = kfold.random_state

    # Divide testing set into epoch-by-epoch validation and final evaluation sets
    X_test, X_val, y_test, y_val, weights_test, weights_val, mcWeights_test, mcWeights_val, Z_test, Z_val = \
        train_test_split(X_test, y_test, weights_test, mcWeights_test, Z_test, test_size=0.5,
                         random_state=random_state)

    # Convert labels to categorical (needed for multiclass training)
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)

    # Split X into track, MSeg, and constit inputs and reshape dataframes into shape expected by Keras
    # This is an ordered array, so each input is formatted as number of constituents x number of variables
    print("\nPreparing train, test, and validate data for model...\n")
    print("\nPreparing constit data...")
    X_train_constit, X_val_constit, X_test_constit = constit_input.extract_and_split_data(X_train, X_val, X_test,
                                                                                          'clus_pt_0', 'clus_time_')
    print("\nPreparing track data...")
    X_train_track, X_val_track, X_test_track = track_input.extract_and_split_data(X_train, X_val, X_test,
                                                                                  'nn_track_pt_0', 'nn_track_SCTHits_')
    print("\nPreparing MSeg data...")
    X_train_MSeg, X_val_MSeg, X_test_MSeg = MSeg_input.extract_and_split_data(X_train, X_val, X_test,
                                                                              'nn_MSeg_etaPos_0', 'nn_MSeg_t0_')
    print("\nPreparing jet data...")
    X_train_jet, X_val_jet, X_test_jet = jet_input.extract_and_split_data(X_train, X_val, X_test, 'jet_pt', 'jet_phi')
    # Done preparing inputs for model!!
    print("\nDone preparing data for model!!!\n")

    # Now to setup ML architecture
    print("\nSetting up model architecture...\n")
    model = setup_model_architecture(constit_input, track_input, MSeg_input, jet_input, X_train_constit, X_train_track,
                                     X_train_MSeg, X_train_jet, reg_value, hidden_fraction, learning_rate,
                                     dropout_value)
    # Save model configuration for evaluation step
    model.save('keras_outputs/' + dir_name + '/model.h5')  # creates a HDF5 file
    # Show summary of model architecture
    print(model.summary())
    # plot model architecture
    if plt_model:
        plot_model(model, show_shapes=True, to_file='plots/' + dir_name + '/model.png')

    # Setup training inputs, outputs, and weights
    x_to_train = [X_train_constit, X_train_track, X_train_MSeg, X_train_jet.values]
    y_to_train = [y_train, y_train, y_train, y_train, y_train]
    weights_to_train = [weights_train.values, weights_train.values, weights_train.values, weights_train.values,
                        weights_train.values]
    # Setup validation inputs, outputs, and weights
    x_to_validate = [X_val_constit, X_val_track, X_val_MSeg, X_val_jet.values]
    y_to_validate = [y_val, y_val, y_val, y_val, y_val]
    weights_to_validate = [weights_val.values, weights_val.values, weights_val.values, weights_val.values,
                           weights_val.values]
    # Setup testing input, outputs, and weights
    x_to_test = [X_test_constit, X_test_track, X_test_MSeg, X_test_jet.values]
    weights_to_test = [weights_test.values, weights_test.values, weights_test.values, weights_test.values,
                       weights_test.values]

    # Do training
    print("\nStarting training...\n")
    validation_data = (x_to_validate, y_to_validate, weights_to_validate)
    callbacks = [EarlyStopping(verbose=True, patience=20, monitor='val_main_output_loss'),
                 ModelCheckpoint('keras_outputs/' + dir_name + '/checkpoint', monitor='val_main_output_loss',
                                 verbose=True, save_best_only=True)]
    history = model.fit(x_to_train, y_to_train, sample_weight=weights_to_train, epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data, callbacks=callbacks)
    # Save model weights
    model.save_weights('keras_outputs/' + dir_name + '/model_weights.h5')

    # Plot training & validation accuracy values
    print("\nPlotting training and validation plots...\n")
    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    plt.plot(history.history['main_output_acc'])
    plt.plot(history.history['val_main_output_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("plots/" + dir_name + "/accuracy_monitoring" + str(n_folds) + ".pdf", format="pdf", transparent=True)
    # Clear axes, figure, and figure window
    plt.clf()
    plt.cla()
    plt.figure()
    # Plot training & validation loss values
    plt.plot(history.history['main_output_loss'])
    plt.plot(history.history['val_main_output_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("plots/" + dir_name + "/loss_monitoring" + str(n_folds) + ".pdf", format="pdf", transparent=True)
    # close all open figures
    plt.close('all')
    del model  # deletes the existing model

    # initialize model with same architecture
    model = load_model('keras_outputs/' + dir_name + '/model.h5')
    # load weights
    model.load_weights('keras_outputs/' + dir_name + '/checkpoint')

    # Evaluate Model with ROC curves
    print("\nEvaluating model...\n")
    # TODO: improve doc on Z and mcWeights
    roc_auc, test_acc = evaluate_model(model, dir_name, x_to_test, y_test, weights_to_test, Z_test, mcWeights_test)
    print('ROC area under curve: %.3f' % roc_auc)
    print('Model accuracy: %.3f' % test_acc)

    return roc_auc, test_acc


def setup_model_architecture(constit_input, track_input, MSeg_input, jet_input, X_train_constit, X_train_track,
                             X_train_MSeg, X_train_jet, reg_value, hidden_fraction, learning_rate, dropout_value):
    """Method that builds the model architecture and returns the model object"""
    # Set up inputs and outputs for Keras layers
    # This sets up the layers specified in the ModelInput object i.e. Conv1D, LSTM
    constit_input_tensor, constit_output_tensor, constit_dense_tensor = constit_input.init_keras_layers(X_train_constit[0].shape, reg_value)
    track_input_tensor, track_output_tensor, track_dense_tensor = track_input.init_keras_layers(X_train_track[0].shape, reg_value)
    MSeg_input_tensor, MSeg_ouput_tensor, MSeg_dense_tensor = MSeg_input.init_keras_layers(X_train_MSeg[0].shape, reg_value)

    # Set up layers for jet
    jet_input_tensor, jet_output_tensor = jet_input.init_keras_dense_input_output(X_train_jet.values[0].shape)
    # Setup concatenation layer
    concat_tensor = concatenate([constit_output_tensor, track_output_tensor, MSeg_ouput_tensor, jet_input_tensor])
    # Setup Dense + Dropout layers
    concat_tensor = Dense(hidden_fraction * 512, activation='relu')(concat_tensor)
    concat_tensor = Dropout(dropout_value)(concat_tensor)
    concat_tensor = Dense(hidden_fraction * 64, activation='relu')(concat_tensor)
    concat_tensor = Dropout(dropout_value)(concat_tensor)
    # Setup final layer
    main_output_tensor = Dense(3, activation='softmax', name='main_output')(concat_tensor)

    # Setup training layers
    layers_to_input = [constit_input_tensor, track_input_tensor, MSeg_input_tensor, jet_input_tensor]
    layers_to_output = [main_output_tensor, constit_dense_tensor, track_dense_tensor,
                        MSeg_dense_tensor, jet_output_tensor]
    weights_for_loss = [1., 0.01, 0.4, 0.1, 0.01]  # TODO: ??

    # Setup Model
    model = Model(inputs=layers_to_input, outputs=layers_to_output)

    # Setup optimiser (Nadam is good as it has decaying learning rate)
    optimizer = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    # Compile Model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', loss_weights=weights_for_loss,
                  metrics=['accuracy'])

    return model


def keras_setup():
    """Sets up Keras"""
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=64, inter_op_parallelism_threads=64)
    tf.set_random_seed(1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)
