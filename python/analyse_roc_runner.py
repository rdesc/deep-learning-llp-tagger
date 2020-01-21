

from analyse_roc_curves import *

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

#roc_files_signalTest = glob.glob("/home/fcormier/calRatio/fullRun2/llp_NN/python/plots/lstm_traininDetails_benchmark/*/training_details.txt")

#print(roc_files_signalTest)

#analyse_roc_signalBenchmark(roc_files_signalTest,"signalTest")

#roc_files_distanceTest = ["lstm_fullSignal_radiusScan_processed_output_Lxy500_Lz500_fracEvents_1.0_constits_30_tracks_20_MSegs_30_LSTMconstits_60_LSTMtracks_60_LSTMmseg_25_kernelReg_0.001_epochs_50_dropout_0.1_doTrackLSTM_True_doMSegLSTM_True_2019-12-11_09:25:16/", "lstm_fullSignal_radiusScan_processed_output_Lxy800_Lz1500_fracEvents_1.0_constits_30_tracks_20_MSegs_30_LSTMconstits_60_LSTMtracks_60_LSTMmseg_25_kernelReg_0.001_epochs_50_dropout_0.1_doTrackLSTM_True_doMSegLSTM_True_2019-12-10_19:43:46/", "lstm_fullSignal_radiusScan_processed_output_Lxy1000_Lz2000_fracEvents_1.0_constits_30_tracks_20_MSegs_30_LSTMconstits_60_LSTMtracks_60_LSTMmseg_25_kernelReg_0.001_epochs_50_dropout_0.1_doTrackLSTM_True_doMSegLSTM_True_2019-12-10_15:58:49/", "lstm_fullSignal_radiusScan_processed_output_Lxy1200_Lz2500_fracEvents_1.0_constits_30_tracks_20_MSegs_30_LSTMconstits_60_LSTMtracks_60_LSTMmseg_25_kernelReg_0.001_epochs_50_dropout_0.1_doTrackLSTM_True_doMSegLSTM_True_2019-12-10_13:40:13/", "lstm_fullSignal_radiusScan_processed_output_Lxy1500_Lz3000_fracEvents_1.0_constits_30_tracks_20_MSegs_30_LSTMconstits_60_LSTMtracks_60_LSTMmseg_25_kernelReg_0.001_epochs_50_dropout_0.1_doTrackLSTM_True_doMSegLSTM_True_2019-12-10_10:09:09/"]
#filename = '/fast_scratch/fcormier/calRatio/fullSignal_withID/processed_output_Lxy500_Lz500.pkl'

#analyse_roc_distanceTest(roc_files_distanceTest, filename)

roc_files_depthTest = glob.glob("/home/fcormier/calRatio/fullRun2/gpu_llpnn/python/plots/lstm_fullSignal_decFinal**/training_details.txt")

analyse_roc_hiddenDepthTest(roc_files_depthTest)

