import argparse
import gc
from deepJet_train_keras import *
from model_input.jet_input import JetInput
from model_input.model_input import ModelInput

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--file_name')
parser.add_argument('--finalPlots_model')
parser.add_argument('--model_to_do')
parser.add_argument('--doTraining', action="store_true")
parser.add_argument('--useGPU2', action="store_true")
parser.add_argument('--makeFinalPlots', action="store_true")
args = parser.parse_args(['--file_name', 'foo', '@args.txt'])

if args.doTraining:
    # Initialize input objects
    constit_input = ModelInput(name='constit', rows_max=30, num_features=12, layers_cnn=[64, 32, 32, 8], nodes_lstm=60)
    track_input = ModelInput(name='track', rows_max=20, num_features=13, layers_cnn=[64, 32, 32, 8], nodes_lstm=60)
    MSeg_input = ModelInput(name='MSeg', rows_max=30, num_features=6, layers_cnn=[32, 16, 4], nodes_lstm=25)
    jet_input = JetInput(name='jet', num_features=3)

    # Train model
    train_llp(args.file_name, args.model_to_do, args.useGPU2, constit_input, track_input, MSeg_input, jet_input, frac=1.0)
    # Free up some memory
    gc.collect()
