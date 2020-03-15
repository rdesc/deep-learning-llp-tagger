import argparse
import gc
import shutil
from datetime import datetime
from deepJet_train_keras import *
from make_final_plots import *
from model_input.jet_input import JetInput
from model_input.model_input import ModelInput
from sklearn.model_selection import StratifiedKFold
from utils import process_kfold_run

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--file_name')
parser.add_argument('--finalPlots_model')
parser.add_argument('--doTraining', action="store_true")
parser.add_argument('--useGPU2', action="store_true")
parser.add_argument('--makeFinalPlots', action="store_true")
parser.add_argument('--doKFold', action="store_true")
parser.add_argument('--doGridSearch', action="store_true")
args = parser.parse_args(['--file_name', 'foo', '@args.txt'])
args = parser.parse_args(['--finalPlots_model', 'foo', '@args.txt'])

# dataset names
name_list = ["processed_output_Lxy1500_Lz3000.pkl"]
# model names
model_to_do_list = ["conv1D_lstm_"]

# model hyper-parameters (can do consecutive training with diff architectures)
filters_cnn_constit = [[64, 32, 32, 16], [64, 32, 32, 12], [64, 32, 32, 8]]
filters_cnn_track = [[64, 32, 32, 16], [64, 32, 32, 12], [64, 32, 32, 8]]
filters_cnn_MSeg = [[32, 16, 8], [32, 16, 6], [32, 16, 4]]
nodes_lstm_constit = [60, 60, 60]
nodes_track_constit = [60, 60, 60]
nodes_MSeg_constit = [25, 25, 25]

# other hyper-parameters
lr_values = [0.00005, 0.000025, 0.0001, 0.0002, 0.0004]
frac_list = [0.2, 0.4, 0.6, 0.8]
layers_list = [1, 2]
node_list = [150, 300]
hidden_layer_fractions = [1, 2, 3]
reg_values = [0.005, 0.0025, 0.001, 0.01]

# KFold variables
kfold = None
roc_results = []
acc_results = []
model_files = []
if args.doKFold:
    # Setup KFold Cross Validation
    seed = np.random.randint(100)  # ensures random shuffling same across models
    n_folds = 5  # should be greater than 2 (usually around 5 - 10 folds)
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

if args.doTraining:
    # iterate over each dataset
    for file in name_list:
        file_name = args.file_name + file

        for model in model_to_do_list:
            # grid search
            for lr in lr_values:
                for reg in reg_values:
                    for i in range(len(filters_cnn_constit)):
                        model_to_do = model + "lr=" + str(lr) + ",reg=" + str(reg) + ",fconv=" + \
                                      str(filters_cnn_constit[i][3])

                        print("\nModel: " + model_to_do)

                        # Initialize input objects
                        constit_input = ModelInput(name='constit', rows_max=30, num_features=12,
                                                   filters_cnn=filters_cnn_constit[i],
                                                   nodes_lstm=nodes_lstm_constit[i])
                        track_input = ModelInput(name='track', rows_max=20, num_features=13,
                                                 filters_cnn=filters_cnn_track[i],
                                                 nodes_lstm=nodes_track_constit[i])
                        MSeg_input = ModelInput(name='MSeg', rows_max=30, num_features=6,
                                                filters_cnn=filters_cnn_MSeg[i],
                                                nodes_lstm=nodes_MSeg_constit[i])
                        jet_input = JetInput(name='jet', num_features=3)

                        # Train model
                        # return dirname to put in same dir as kfold
                        roc_scores, acc_scores, dir_name = train_llp(file_name, model_to_do, args.useGPU2,
                                                                     constit_input, track_input,
                                                                     MSeg_input, jet_input, frac=1.0,
                                                                     plt_model=True, learning_rate=lr,
                                                                     hidden_fraction=2, epochs=100,
                                                                     dropout_value=0.2,
                                                                     reg_value=reg, kfold=kfold)

                        # Summarize performance metrics
                        print('\nEstimated AUC %.3f (%.3f)' % (np.mean(roc_scores), np.std(roc_scores)))
                        print('Estimated accuracy %.3f (%.3f)' % (np.mean(acc_scores), np.std(acc_scores)))
                        roc_results.append(roc_scores)
                        acc_results.append(acc_scores)
                        model_files.append(dir_name)

                        # Free up some memory
                        gc.collect()

    # Make boxplots of kFold CV results
    if args.doKFold:
        process_kfold_run(roc_results, acc_results, model_to_do_list, model_files, name_list, seed)

    # Put all model files in the same directory
    if args.doGridSearch:
        creation_time = str(datetime.now().strftime('%m-%d_%H:%M'))
        gridSearch_dir = "gridSearch_" + creation_time
        for f in model_files:
            shutil.move("plots/" + f, "plots/" + gridSearch_dir + "/" + f)

        # aggregate model metrics and reorder lists in decreasing performance order
        roc_results = np.asarray(roc_results)
        acc_results = np.asarray(acc_results)
        model_files = np.asarray(model_files)
        order = np.argsort(-1*roc_results)
        roc_results = roc_results[order]
        acc_results = acc_results[order]
        model_files = model_files[order]

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

if args.makeFinalPlots:
    input_file = args.file_name + "/validation_dec24.pkl"
    plot_vars_final(input_file, model_to_do=args.finalPlots_model, num_constit_lstm=150, num_track_lstm=150,
                    num_mseg_lstm=150, learning_rate=0.00005, numConstitLayers=1, numTrackLayers=1, numMSegLayers=1,
                    hiddenFraction=2, epochs=200, dropout_value=0.2, reg_value=0.002),
