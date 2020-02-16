import pandas as pd
import numpy as np
import os
from datetime import datetime
from keras.models import load_model
import matplotlib.pyplot as plt
from evaluate_training import find_threshold, signal_llp_efficiencies, bkg_falsePositives,\
    make_multi_roc_curve, plot_prediction_histograms


def create_directories(model_to_do, filename):
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


def evaluate_model(model, dir_name, X, y, Z, mcWeights):
    # TODO: refactor
    # make predictions
    prediction = model.predict(X, verbose=True)
    prediction = prediction[0]  # TODO check

    # Sum of MC weights
    bib_weight = np.sum(mcWeights[y == 2])
    sig_weight = np.sum(mcWeights[y == 1])
    qcd_weight = np.sum(mcWeights[y == 0])

    bib_weight_length = len(mcWeights[y == 2])
    sig_weight_length = len(mcWeights[y == 1])
    qcd_weight_length = len(mcWeights[y == 0])

    mcWeights[y == 0] *= qcd_weight_length / qcd_weight
    mcWeights[y == 2] *= bib_weight_length / bib_weight
    mcWeights[y == 1] *= sig_weight_length / sig_weight
    destination = "plots/" + dir_name + "/"
    plot_prediction_histograms(destination, prediction, y, mcWeights, dir_name)

    # This will be the BIB efficiencies to aim for when making family of ROC curves
    threshold_array = [(1 - 0.0316)]
    counter = 0
    # Third label: the label of the class we are doing a 'family' of. Other two classes will make the ROC curve
    third_label = 2
    # We'll be writing the stats to training_details.txt
    f = open(destination + "training_details.txt", "a")
    f.write("\nEvaluation metrics\n")

    # Loop over all arrays in threshold_array
    for item in threshold_array:
        # Convert decimal to percentage (code used was in percentage, oh well)
        test_perc = item * 100
        test_label = third_label

        # Find threshold, or at what label we will have the required percentage of 'test_label' correctl predicted
        test_threshold, leftovers = find_threshold(prediction, y, mcWeights, test_perc, test_label)
        # Make ROC curve of leftovers, those not tagged by above function
        bkg_eff, tag_eff, roc_auc = make_multi_roc_curve(prediction, y, mcWeights, test_threshold, test_label,
                                                         leftovers)
        # Write AUC to training_details.txt
        f.write("%s, %s\n" % (str(-item + 1), str(roc_auc)))
        print("AUC: " + str(roc_auc))
        # Make ROC curve
        plt.plot(tag_eff, bkg_eff, label=f"BIB Eff: {item :.3f}" + f", AUC: {roc_auc:.3f}")
        plt.xlabel("LLP Tagging Efficiency")
        axes = plt.gca()
        axes.set_xlim([0, 1])
        counter = counter + 1

    # Finish and plot ROC curve family
    plt.legend()
    plt.yscale('log', nonposy='clip')
    signal_test = prediction[y == 1]
    qcd_test = prediction[y == 0]

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
    plt.cla()
    # Make plots of signal efficiency vs mH, mS
    signal_llp_efficiencies(prediction, y, Z, destination, f)
    bkg_falsePositives(prediction, y, Z, destination, f)
    f.close()

