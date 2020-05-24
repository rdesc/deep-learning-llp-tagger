import argparse
import gc
import numpy as np
from deepJet_train_keras import train_llp
from model_input.jet_input import JetInput
from model_input.model_input import ModelInput
from sklearn.model_selection import StratifiedKFold
from utils import process_kfold_run, process_grid_search_run

# parse arguments from args.txt
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--data_dir')
parser.add_argument('--useGPU2', action="store_true")
parser.add_argument('--doKFold', action="store_true")
parser.add_argument('--doGridSearch', action="store_true")
args = parser.parse_args(['--data_dir', 'foo', '@args.txt'])

# specify file names that are inside the directory "--data_dir"
file_names = ["processed_output_Lxy1500_Lz3000.pkl"]
# names to label models
model_names = ["lstm_", "conv1D_", "lstm_conv1D_"]

# model hyperparameters
filters_cnn_constit = [0, [64, 32, 32, 8], [64, 32, 32, 8]]
filters_cnn_track = [0, [64, 32, 32, 8], [64, 32, 32, 8]]
filters_cnn_MSeg = [0, [32, 16, 4], [32, 16, 4]]
nodes_lstm_constit = [60, 0, 60]
nodes_track_constit = [60, 0, 60]
nodes_MSeg_constit = [25, 0, 25]

# other hyperparameters
lr_values = [0.00005, 0.000025, 0.0001, 0.0002, 0.0004]
frac_list = [0.2, 0.4, 0.6, 0.8]
layers_list = [1, 2]
node_list = [150, 300]
hidden_layer_fractions = [1, 2, 3]
reg_values = [0.005, 0.0025, 0.001, 0.01]


def init_training(file_name, model, constit_input, track_input, MSeg_input, jet_input, lr=0.0004, reg=0.001, kfold=None):
    """
    Helper function to start training
    """
    roc_scores, acc_scores, dir_name = train_llp(file_name, model, args.useGPU2, constit_input, track_input, MSeg_input,
                                                 jet_input, frac=1.0, plt_model=True, learning_rate=lr, hidden_fraction=2,
                                                 epochs=100, dropout_value=0.2, reg_value=reg, kfold=kfold)
    # Summarize performance metrics
    print('\nEstimated AUC %.3f (%.3f)' % (np.mean(roc_scores), np.std(roc_scores)))
    print('Estimated accuracy %.3f (%.3f)' % (np.mean(acc_scores), np.std(acc_scores)))
    # Free up some memory
    gc.collect()

    return roc_scores, acc_scores, dir_name


if __name__ == '__main__':
    # initialize variables to store metrics
    roc_results = []
    acc_results = []
    model_files = []

    # iterate over each dataset
    for file in file_names:
        file_name = args.data_dir + file

        if args.doKFold:
            # Setup KFold Cross Validation
            seed = np.random.randint(100)  # ensures random shuffling same across models
            n_folds = 5  # should be greater than 2 (usually around 5 - 10 folds)
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            for i, model in enumerate(model_names):
                print("\nModel: " + model)

                # initialize model inputs
                constit_input = ModelInput(name='constituent', input_count=30, vars_count=12,
                                           filters_cnn=filters_cnn_constit[i], nodes_lstm=nodes_lstm_constit[i])
                track_input = ModelInput(name='track', input_count=20, vars_count=13,
                                         filters_cnn=filters_cnn_track[i], nodes_lstm=nodes_track_constit[i])
                MSeg_input = ModelInput(name='muon_segment', input_count=30, vars_count=6,
                                        filters_cnn=filters_cnn_MSeg[i], nodes_lstm=nodes_MSeg_constit[i])
                jet_input = JetInput(name='jet', vars_count=3)

                roc_scores, acc_scores, dir_name = init_training(file_name, model, constit_input, track_input, MSeg_input, jet_input, kfold=kfold)
                roc_results.append(roc_scores)
                acc_results.append(acc_scores)
                model_files.append(dir_name)

            # process and save the results of the completed KFold
            process_kfold_run(roc_results, acc_results, model_names, model_files, file_names, seed)

        elif args.doGridSearch:
            # iterate through all the possible hyperparameter configurations
            for model in model_names:
                for lr in lr_values:
                    for reg in reg_values:
                        for i in range(len(filters_cnn_constit)):
                            model_name = model + "lr=" + str(lr) + ",reg=" + str(reg) + ",fconv=" + str(filters_cnn_constit[i])
                            print("\nModel: " + model_name)

                            # Initialize input objects
                            constit_input = ModelInput(name='constituent', input_count=30, vars_count=12,
                                                       filters_cnn=filters_cnn_constit[i], nodes_lstm=nodes_lstm_constit[i])
                            track_input = ModelInput(name='track', input_count=20, vars_count=13,
                                                     filters_cnn=filters_cnn_track[i], nodes_lstm=nodes_track_constit[i])
                            MSeg_input = ModelInput(name='muon_segment', input_count=30, vars_count=6,
                                                    filters_cnn=filters_cnn_MSeg[i], nodes_lstm=nodes_MSeg_constit[i])
                            jet_input = JetInput(name='jet', vars_count=3)

                            roc_scores, acc_scores, dir_name = init_training(file_name, model_name, constit_input, track_input, MSeg_input, jet_input, lr=lr, reg=reg)

                            roc_results.append(roc_scores)
                            acc_results.append(acc_scores)
                            model_files.append(dir_name)

            # process and save the results of the completed Grid Search
            process_grid_search_run(roc_results, acc_results, model_files, lr_values, reg_values, filters_cnn_constit, filters_cnn_track, filters_cnn_MSeg)

        else:
            # no KFold or grid search, just do standard training
            model = "lstm_conv1D_"
            print("\nModel: " + model)

            # Initialize input objects
            constit_input = ModelInput(name='constituent', input_count=30, vars_count=12, filters_cnn=[64, 32, 32, 8], nodes_lstm=60)
            track_input = ModelInput(name='track', input_count=20, vars_count=13, filters_cnn=[64, 32, 32, 8], nodes_lstm=60)
            MSeg_input = ModelInput(name='muon_segment', input_count=30, vars_count=6, filters_cnn=[32, 16, 4], nodes_lstm=25)
            jet_input = JetInput(name='jet', vars_count=3)

            init_training(file_names[0], model, constit_input, track_input, MSeg_input, jet_input)
