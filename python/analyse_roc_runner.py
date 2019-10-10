

from analyze_roc import *

#roc_files_benchmark = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/ml1_results/benchmark/benchmark/lstm_*/training_details.txt")

#analyse_roc_benchmark(roc_files_benchmark)

#roc_files_frac = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_fracTest/lstm_*/training_details.txt")

#analyse_roc_frac(roc_files_frac)

#roc_files_numConstits = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_numConstitsTest/lstm_numConstitsTest/lstm_*/training_details.txt")

#analyse_roc_numMaxConstits(roc_files_numConstits)

#roc_files_numTracks = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_numTracksTest/lstm*/training_details.txt")

#analyse_roc(roc_files_numTracks,2,"numMaxTracks")

#roc_files_constitLSTM = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_numConstitLSTM/lstm*/training_details.txt")

#analyse_roc(roc_files_constitLSTM,4,"constitLSTM")

#roc_files_trackLSTM = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_numTrackLSTM/lstm*/training_details.txt")

#analyse_roc(roc_files_trackLSTM,5,"trackLSTM")

#roc_files_msegLSTM = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_numMSegLSTM/lstm*/training_details.txt")

#roc_files_inclusionTest = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_inclusionTest/*/training_details.txt")

#print(roc_files_inclusionTest)

#analyse_roc_inclusion(roc_files_inclusionTest,6,"inclusionTest")

roc_files_signalTest = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_traininDetails_benchmark/*/training_details.txt")

print(roc_files_signalTest)

analyse_roc_signalBenchmark(roc_files_signalTest,"signalTest")

