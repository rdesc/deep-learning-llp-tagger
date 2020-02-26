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
# model names
model_to_do_list = ["conv1D_lstm_", "lstm_"]

# model hyper-parameters (can do consecutive training with diff architectures)
filters_cnn_constit = [[64, 32, 32, 8], 0]
filters_cnn_track = [[64, 32, 32, 8], 0]
filters_cnn_MSeg = [[32, 16, 4], 0]
nodes_lstm_constit = [150, 150]
nodes_track_constit = [150, 150]
nodes_MSeg_constit = [150, 150]

if args.doTraining:
    # iterate over each dataset
    for file in name_list:
        file_name = args.file_name + file

        for i in range(len(model_to_do_list)):
            model_to_do = model_to_do_list[i]
            print("\nModel: " + model_to_do)

            # Initialize input objects
            constit_input = ModelInput(name='constit', rows_max=30, num_features=12, filters_cnn=filters_cnn_constit[i],
                                       nodes_lstm=nodes_lstm_constit[i])
            track_input = ModelInput(name='track', rows_max=20, num_features=13, filters_cnn=filters_cnn_track[i],
                                     nodes_lstm=nodes_track_constit[i])
            MSeg_input = ModelInput(name='MSeg', rows_max=30, num_features=6, filters_cnn=filters_cnn_MSeg[i],
                                    nodes_lstm=nodes_MSeg_constit[i])
            jet_input = JetInput(name='jet', num_features=3)

            # Train model
            train_llp(file_name, model_to_do, args.useGPU2, constit_input, track_input, MSeg_input, jet_input, frac=1.0,
                      plt_model=True, learning_rate=0.00005, hidden_fraction=2, epochs=50, dropout_value=0.2,
                      reg_value=0.005)
            # Free up some memory
            gc.collect()

if args.makeFinalPlots:
    input_file = args.file_name + "/validation_dec24.pkl"
    plot_vars_final(input_file, model_to_do=args.finalPlots_model, num_constit_lstm=150, num_track_lstm=150,
                    num_mseg_lstm=150, learning_rate=0.00005, numConstitLayers=1, numTrackLayers=1, numMSegLayers=1,
                    hiddenFraction=2, epochs=200, dropout_value=0.2, reg_value=0.002),

