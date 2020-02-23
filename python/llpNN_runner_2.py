import argparse
import gc
from deepJet_train_keras import *
from make_final_plots import *
from model_input.jet_input import JetInput
from model_input.model_input import ModelInput

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--file_name')
parser.add_argument('--finalPlots_model')
parser.add_argument('--doTraining', action="store_true")
parser.add_argument('--useGPU2', action="store_true")
parser.add_argument('--makeFinalPlots', action="store_true")
args = parser.parse_args(['--file_name', 'foo', '@args.txt'])
args = parser.parse_args(['--finalPlots_model', 'foo', '@args.txt'])

# dataset names
name_list = ["processed_output_Lxy1500_Lz3000_slim0.1.pkl"]
model_to_do = "conv1d_"

# model hyper-parameters (can do consecutive training with diff architectures)
# number of nodes in LSTM for each input
constit_nodes_lstm = [60, 120, 240]
track_nodes_lstm = [60, 120, 240]
MSeg_nodes_lstm = [25, 50, 200]
frac_list = [1.0, 0.2, 0.4, 0.6, 0.8]
lr_values = [0.00005]

# number of inputs
num_constits_list = [30, 28, 26, 22, 16, 12, 8]
num_tracks_list = [20, 15, 10, 5]

# other parameters
layers_list = [1, 2]
node_list = [150, 300]

if args.doTraining:
    # iterate over each dataset
    for file in name_list:
        file_name = args.file_name + file

        # TODO: add loop to iterate over each model architecture
        # Initialize input objects
        constit_input = ModelInput(name='constit', rows_max=30, num_features=12, filters_cnn=[64, 32, 32, 8], nodes_lstm=60)
        track_input = ModelInput(name='track', rows_max=20, num_features=13, filters_cnn=[64, 32, 32, 8], nodes_lstm=60)
        MSeg_input = ModelInput(name='MSeg', rows_max=30, num_features=6, filters_cnn=[32, 16, 4], nodes_lstm=25)
        jet_input = JetInput(name='jet', num_features=3)

        # Train model
        train_llp(file_name, model_to_do, args.useGPU2, constit_input, track_input, MSeg_input, jet_input, frac=1.0)
        # Free up some memory
        gc.collect()

if args.makeFinalPlots:
    input_file = args.file_name + "/validation_dec24.pkl"
    plot_vars_final(input_file, model_to_do=args.finalPlots_model, num_constit_lstm=150, num_track_lstm=150,
                    num_mseg_lstm=150, learning_rate=0.00005, numConstitLayers=1, numTrackLayers=1, numMSegLayers=1,
                    hiddenFraction=2, epochs=200, dropout_value=0.2, reg_value=0.002),
