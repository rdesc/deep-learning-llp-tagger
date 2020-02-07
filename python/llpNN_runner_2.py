import argparse
import gc
from deepJet_train_keras import train_llp
from track_input import TrackInput
from constit_input import ConstitInput
from mseg_input import MSegInput
from jet_input import JetInput

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--file_name')
parser.add_argument('--finalPlots_model')
parser.add_argument('--doTraining', action="store_true")
parser.add_argument('--useGPU2', action="store_true")
parser.add_argument('--makeFinalPlots', action="store_true")
args = parser.parse_args(['--file_name', 'foo', '@args.txt'])

if args.doTraining:
    # Initialize input objects
    constit_input = ConstitInput(rows_max=30, num_features=12, layers_cnn=[64, 32, 32, 8], nodes_lstm=60)
    track_input = TrackInput(rows_max=20, num_features=13, layers_cnn=[64, 32, 32, 8], nodes_lstm=60)
    MSeg_input = MSegInput(rows_max=30, num_features=6, layers_cnn=[32, 16, 4], nodes_lstm=25)
    jet_input = JetInput(num_features=3)

    # Train model
    train_llp(args.file_name, args.useGPU2, constit_input, track_input, MSeg_input, jet_input, frac=1.0)
    # Free up some memory
    gc.collect()
