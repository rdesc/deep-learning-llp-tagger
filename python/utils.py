import pandas as pd
import numpy as np
import os

# TODO: fix
def create_directories(model_to_do):
    """Creates directories for keras outputs and plots

    Parameters
    ----------
    model_to_do : the model
    """
    # Create directories
    dir_name = "plots/" + model_to_do
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        os.mkdir("plots/trainingDiagrams/" + model_to_do)
        os.mkdir("keras_outputs/" + model_to_do)
        print("Directory ", dir_name, " Created")
    else:
        print("Directory ", dir_name, " already exists")

    destination = "plots/" + model_to_do + "/"

    # Write a file with some details of architecture, will append final stats at end of training
    f = open(destination + "training_details.txt", "w+")
    f.write(
        "\nnum_max_constits = %s\nnum_max_tracks = %s\nnum_max_MSegs = %s\nnum_constit_cnn = %s\nnum_track_cnn = %s\nnum_mseg_cnn = %s\n" % (
            num_max_constits, num_max_tracks, num_max_MSegs, num_constit_cnn, num_track_cnn, num_mseg_cnn))
    f.close()

    # Print these stats to stdout
    print(
        "\nnum_max_constits = %s\nnum_max_tracks = %s\nnum_max_MSegs = %s\nnum_constit_cnn = %s\nnum_track_cnn = %s\nnum_mseg_cnn = %s\n" % (
            num_max_constits, num_max_tracks, num_max_MSegs, num_constit_cnn, num_track_cnn, num_mseg_cnn))


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
