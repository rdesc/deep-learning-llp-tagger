import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from evaluate_training import find_threshold, signal_llp_efficiencies, bkg_falsePositives, \
    make_multi_roc_curve, plot_prediction_histograms
from keras.utils import np_utils


def create_directories(model_to_do, filename):
    """Creates directories to store model plots + Keras files and returns directory name."""
    # Append time/date to directory name
    creation_time = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S/'))
    dir_name = model_to_do + filename + "_" + creation_time

    # Create directories
    os.makedirs("plots/" + dir_name)
    print("Directory plots/" + dir_name + " created!")

    os.makedirs("keras_outputs/" + dir_name)
    print("Directory keras_outputs/" + dir_name + " created!")

    return dir_name


def load_dataset(filename):
    """Loads .pkl file, does some pre-processing and returns Pandas DataFrame"""
    # Load dataset
    df = pd.read_pickle(filename)
    # Replace infs with nans
    df = df.replace([np.inf, -np.inf], np.nan)
    # Replace nans with 0
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
    print("\nLength of Signal is: " + str(df[df.label == 1].shape[0]))
    print("Length of QCD is: " + str(df[df.label == 0].shape[0]))
    print("Length of BIB is: " + str(df[df.label == 2].shape[0]))

    return df


def process_kfold_run(roc_results, acc_results, model_to_do_list, model_files, name_list, seed):
    """Generates ROC AUC and accuracy plots from KFold run and saves results to a .txt file"""
    print("\nPlotting KFold Cross Validation results...\n")
    creation_time = str(datetime.now().strftime('%m-%d_%H:%M'))
    kfold_dir = "kfold_" + creation_time

    # create directory for kfold plots
    os.makedirs("plots/" + kfold_dir)
    # move model files to kfold directory
    for f in model_files:
        shutil.move("plots/" + f, "plots/" + kfold_dir + "/" + f)

        # plot roc auc scores
    fig = plt.figure()
    fig.suptitle('Model Comparison with ROC AUC metric')
    ax = fig.add_subplot(111)
    plt.boxplot(roc_results)
    ax.set_xticklabels(model_to_do_list)
    fig.savefig("plots/" + kfold_dir + "/kfold_cv_roc.pdf", format="pdf", transparent=True)

    # plot accuracy scores
    fig = plt.figure()
    fig.suptitle('Model Comparison with accuracy metric')
    ax = fig.add_subplot(111)
    plt.boxplot(acc_results)
    ax.set_xticklabels(model_to_do_list)
    fig.savefig("plots/" + kfold_dir + "/kfold_cv_acc.pdf", format="pdf", transparent=True)

    # save results to file
    f = open("plots/" + kfold_dir + "/kfold_data.txt", "w+")
    f.write("File name list\n")
    f.write(str(name_list))
    f.write("\nModel list\n")
    f.write(str(model_to_do_list))
    f.write("\nROC AUC data\n")
    f.write(str(roc_results))
    f.write("\nAccuracy data\n")
    f.write(str(acc_results))
    f.write("\nSeed\n")
    f.write(str(seed))
    f.close()


def process_grid_search_run(roc_results, acc_results, model_files, lr_values, reg_values, filters_cnn_constit, filters_cnn_track, filters_cnn_MSeg):
    """Aggregates model files and metrics built during grid search"""
    print("\nProcessing grid search results...\n")
    creation_time = str(datetime.now().strftime('%m-%d_%H:%M'))
    gridSearch_dir = "gridSearch_" + creation_time
    print("\nSuccessfully trained %.0f models\n" % len(model_files))
    for f in model_files:
        shutil.move("plots/" + f, "plots/" + gridSearch_dir + "/" + f)

    # aggregate model metrics and reorder lists in decreasing performance order
    roc_results = np.asarray(roc_results)
    acc_results = np.asarray(acc_results)
    model_files = np.asarray(model_files)
    order = np.argsort(-1 * roc_results)
    roc_results = roc_results[order]
    acc_results = acc_results[order]
    model_files = model_files[order]

    # rebuild all the model hyper-parameter configurations
    lr = []
    reg = []
    constit = []
    track = []
    mseg = []
    for l in lr_values:
        for r in reg_values:
            for i in range(len(filters_cnn_track)):
                lr.append(l)
                reg.append(r)
                constit.append(filters_cnn_constit[i][-1])
                track.append(filters_cnn_track[i][-1])
                mseg.append(filters_cnn_MSeg[i][-1])

    # put data into a pandas DataFrame
    columns = ['learning_rate', 'regularization', 'cnn_final_layer_constit', 'cnn_final_layer_track', 'cnn_final_layer_MSeg', 'roc_score', 'acc_score', 'rank']
    data_list = [lr, reg, constit, track, mseg, roc_results, acc_results, order]
    data = {}
    for i in range(len(columns)):
        data[columns[i]] = data_list[i]
    df = pd.DataFrame(data, columns=columns)

    # save df to file
    df.to_csv("plots/" + gridSearch_dir + "/df.csv", index=False)

    # print out correlation matrix
    print(df.corr())

    # save results to file
    f = open("plots/" + gridSearch_dir + "/gridSearch_data.txt", "w+")
    f.write("Model list\n")
    f.write(str(model_files))
    f.write("\n\nROC AUC data\n")
    f.write(str(roc_results))
    f.write("\n\nAccuracy data\n")
    f.write(str(acc_results))
    f.write("\n\nOrder\n")
    f.write(str(order))
    f.write("\n\nLearning values\n")
    f.write(str(lr_values))
    f.write("\n\nRegularization values\n")
    f.write(str(reg_values))
    f.write("\n\nFilters constit\n")
    f.write(str(filters_cnn_constit))
    f.write("\n\nFilters track\n")
    f.write(str(filters_cnn_track))
    f.write("\n\nFilters MSeg\n")
    f.write(str(filters_cnn_MSeg))
    f.close()


def evaluate_model(model, dir_name, X_test, y_test, weights_test, Z_test, mcWeights_test, n_folds):
    # TODO: add doc for method + params, add kfold param

    # evaluate the model using Keras api
    acc_index = model.metrics_names.index('main_output_categorical_accuracy')
    # model.evaluate expects target data to be the same shape/format as model.fit
    y_eval = np_utils.to_categorical(y_test) 
    y_eval = [y_eval, y_eval, y_eval, y_eval, y_eval]
    # get accuracy of model on test set
    test_acc = model.evaluate(X_test, y_eval, verbose=1, sample_weight=weights_test)[acc_index]

    # TODO: refactor and understand
    # make predictions
    prediction = model.predict(X_test, verbose=0)  # currently expects X to be a list of length 4 (model has 4 inputs)
    prediction = prediction[0]  # model currently has 5 outputs (1 main output, 4 outputs for monitoring LSTMs)

    # Sum of MC weights
    bib_weight = np.sum(mcWeights_test[y_test == 2])
    sig_weight = np.sum(mcWeights_test[y_test == 1])
    qcd_weight = np.sum(mcWeights_test[y_test == 0])

    bib_weight_length = len(mcWeights_test[y_test == 2])
    sig_weight_length = len(mcWeights_test[y_test == 1])
    qcd_weight_length = len(mcWeights_test[y_test == 0])

    mcWeights_test[y_test == 0] *= qcd_weight_length / qcd_weight
    mcWeights_test[y_test == 2] *= bib_weight_length / bib_weight  # TODO: this does nothing??
    mcWeights_test[y_test == 1] *= sig_weight_length / sig_weight
    destination = "plots/" + dir_name + "/"
    # TODO: to add other plots when nfold?
    plot_prediction_histograms(destination, prediction, y_test, mcWeights_test, dir_name)

    # This will be the BIB efficiency to aim for when making ROC curve
    threshold = 1 - 0.0316
    # Third label: the label of the class we are doing a 'family' of. Other two classes will make the ROC curve
    third_label = 2
    # We'll be writing the stats to training_details.txt
    f = open(destination + "training_details.txt", "a")
    if n_folds:
        f.write("KFold iteration # %s" % str(n_folds))
    f.write("\nEvaluation metrics\n")

    # Find threshold, or at what label we will have the required percentage of 'test_label' correctl predicted
    test_threshold, leftovers = find_threshold(prediction, y_test, mcWeights_test, threshold * 100, third_label)
    # Make ROC curve of leftovers, those not tagged by above function
    bkg_eff, tag_eff, roc_auc = make_multi_roc_curve(prediction, y_test, mcWeights_test, test_threshold, third_label,
                                                     leftovers)
    # TODO: uncomment rest 
    # Write AUC to training_details.txt
    f.write("Threshold: %s, ROC AUC: %s\n" % (str(-threshold + 1), str(roc_auc)))
    f.write("Accuracy: %s\n" % str(test_acc))
    print("AUC: " + str(roc_auc))
    # Make ROC curve
    plt.plot(tag_eff, bkg_eff, label=f"BIB Eff: {threshold :.3f}" + f", AUC: {roc_auc:.3f}")
    plt.xlabel("LLP Tagging Efficiency")
    axes = plt.gca()
    axes.set_xlim([0, 1])

    # Finish and plot ROC curve family
    plt.legend()
    plt.yscale('log', nonposy='clip')
    signal_test = prediction[y_test == 1]
    qcd_test = prediction[y_test == 0]

    print(signal_test[0:100].shape)
    print("Length of Signal: " + str(len(signal_test)) + ", length of signal with weight 1: " + str(
        len(signal_test[signal_test[:, 1] < 0.1])))
    print("Length of QCD: " + str(len(qcd_test)) + ", length of qcd with weight 1: " + str(
        len(qcd_test[qcd_test[:, 1] < 0.1])))
    if third_label == 2:
        plt.ylabel("QCD Rejection")
        plt.savefig(destination + "roc_curve_atlas_rej_bib" + ".pdf", format='pdf', transparent=True)
    if third_label == 0:
        plt.ylabel("BIB Rejection")
        plt.savefig(destination + "roc_curve_atlas_rej_qcd" + ".pdf", format='pdf', transparent=True)
    plt.clf()
    # Make plots of signal efficiency vs mH, mS
    signal_llp_efficiencies(prediction, y_test, Z_test, destination, f)
    bkg_falsePositives(prediction, y_test, Z_test, destination, f)
    f.close()

    return roc_auc, test_acc
