import argparse
import gc
from deepJet_train_keras import *
from make_final_plots import *
from model_input.jet_input import JetInput
from model_input.model_input import ModelInput
from sklearn.model_selection import KFold
from datetime import datetime

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--file_name')
parser.add_argument('--finalPlots_model')
parser.add_argument('--doTraining', action="store_true")
parser.add_argument('--useGPU2', action="store_true")
parser.add_argument('--makeFinalPlots', action="store_true")
parser.add_argument('--doKFold', action="store_true")
args = parser.parse_args(['--file_name', 'foo', '@args.txt'])
args = parser.parse_args(['--finalPlots_model', 'foo', '@args.txt'])

# dataset names
name_list = ["processed_output_Lxy1500_Lz3000.pkl"]
# model names
model_to_do_list = ["conv1D_lstm_", "lstm_", "conv1D_"]

# model hyper-parameters (can do consecutive training with diff architectures)
filters_cnn_constit = [[64, 32, 32, 8], 0, [64, 32, 32, 8]]
filters_cnn_track = [[64, 32, 32, 8], 0, [64, 32, 32, 8]]
filters_cnn_MSeg = [[32, 16, 4], 0, [32, 16, 4]]
nodes_lstm_constit = [150, 150, 0]
nodes_track_constit = [150, 150, 0]
nodes_MSeg_constit = [150, 150, 0]

# other hyper-parameters
num_constits_lstm = [60, 120, 240]
num_tracks_lstm = [60, 120, 240]
num_msegs_lstm = [25, 50, 200]
lr_values = [0.00005]
frac_list = [0.2, 0.4, 0.6, 0.8]
layers_list = [1, 2]
node_list = [150, 300]
hidden_layer_fractions = [2, 3, 4]

kfold = None
roc_results = []
acc_results = []
if args.doKFold:
    # Setup KFold Cross Validation
    seed = np.random.randint(100)
    n_folds = 5
    kfold = KFold(n_folds, True, seed)

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
            roc_scores, acc_scores = train_llp(file_name, model_to_do, args.useGPU2, constit_input, track_input,
                                               MSeg_input, jet_input, frac=1.0,
                                               plt_model=True, learning_rate=0.00005, hidden_fraction=2, epochs=50,
                                               dropout_value=0.2,
                                               reg_value=0.005, kfold=kfold)

            # Summarize performance metrics
            print('Estimated AUC %.3f (%.3f)' % (np.mean(roc_scores), np.std(roc_scores)))
            print('Estimated accuracy %.3f (%.3f)' % (np.mean(acc_scores), np.std(acc_scores)))
            roc_results.append(roc_scores)
            acc_results.append(acc_scores)

            # TODO: save results in a .txt file somewhere
            # Free up some memory
            gc.collect()

    # Make boxplots of kFold CV
    if roc_results and acc_results:
        # save results to file first
        
        print("\nPlotting KFold Cross Validation results...\n")
        creation_time = str(datetime.now().strftime('%m-%d_%H:%M'))

        # plot roc auc scores
        fig = plt.figure()
        fig.suptitle('Model Comparison with ROC AUC metric')
        ax = fig.add_subplot(111)
        plt.boxplot(roc_results)
        ax.set_xticklabels(model_to_do_list)
        fig.savefig("plots/kfold_cv_roc_" + creation_time + ".pdf", format="pdf", transparent=True)

        # plot accuracy scores
        fig = plt.figure()
        fig.suptitle('Model Comparison with accuracy metric')
        ax = fig.add_subplot(111)
        plt.boxplot(acc_results)
        ax.set_xticklabels(model_to_do_list)
        fig.savefig("plots/kfold_cv_acc_" + creation_time + ".pdf", format="pdf", transparent=True)

        f = open("plots/kfold_data_" + creation_time + ".txt", "w+")
        f.write("ROC\n")
        f.write(str(roc_results))
        f.write("\nACC\n")
        f.write(str(acc_results))
        f.close()

if args.makeFinalPlots:
    input_file = args.file_name + "/validation_dec24.pkl"
    plot_vars_final(input_file, model_to_do=args.finalPlots_model, num_constit_lstm=150, num_track_lstm=150,
                    num_mseg_lstm=150, learning_rate=0.00005, numConstitLayers=1, numTrackLayers=1, numMSegLayers=1,
                    hiddenFraction=2, epochs=200, dropout_value=0.2, reg_value=0.002),
