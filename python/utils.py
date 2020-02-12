import pandas as pd
import numpy as np
import os
from datetime import datetime


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
