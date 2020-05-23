import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

matplotlib.use('agg')


# Function to find threshold weight at which you get percentage (perc) right
def find_threshold(prediction, y, perc, label):
    # Instead of lame loops let's order our data, then find percentage from there
    # prediction is 3xN, want to sort by BIB weight

    label_events_y = y[y == label]
    label_events_prediction = prediction[y == label]

    prediction_sorted = np.array(label_events_prediction[label_events_prediction[:, label].argsort()])

    cutoffIndex = round(((100 - perc) / 100) * label_events_y.size)
    print("CutoffIndex: " + str(int(cutoffIndex)))
    threshold = prediction_sorted.item((int(cutoffIndex), label))
    print("Treshold: " + str(threshold))

    leftovers = np.where(
        np.greater(
            threshold,
            prediction[:, label]))

    return threshold, leftovers


# Plot signal efficiency as function of mH, mS
def signal_llp_efficiencies(prediction, y_test, Z_test, destination, f):
    sig_rows = np.where(y_test == 1)
    prediction = prediction[sig_rows]
    Z_test = Z_test.iloc[sig_rows]
    mass_array = (Z_test.groupby(['llp_mH', 'llp_mS']).size().reset_index().rename(columns={0: 'count'}))
    print(mass_array)

    plot_x = []
    plot_y = []
    plot_z = []

    for item, mH, mS in zip(mass_array['count'], mass_array['llp_mH'], mass_array['llp_mS']):
        temp_array = prediction[(Z_test['llp_mH'] == mH) & (Z_test['llp_mS'] == mS)]
        temp_max = np.argmax(temp_array, axis=1)
        temp_num_signal_best = len(temp_max[temp_max == 1])
        temp_eff = temp_num_signal_best / temp_array.shape[0]
        plot_x.append(mH)
        plot_y.append(temp_eff)
        plot_z.append(mS)
        print("mH: " + str(mH) + ", mS: " + str(mS) + ", Eff: " + str(temp_eff))
        f.write("%s,%s,%s\n" % (str(mH), str(mS), str(temp_eff)))

    plt.clf()
    plt.figure()
    plt.scatter(plot_x, plot_y, marker='+', s=150, linewidths=4, c=plot_z, cmap=plt.cm.coolwarm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'mS')
    plt.xlabel("mH")
    plt.ylabel("Signal Efficiency")

    plt.savefig(destination + "signal_llp_efficiencies" + ".pdf", format='pdf', transparent=True)
    plt.clf()


def bkg_falsePositives(prediction, y_test, Z_test, destination, f):
    qcd_rows = np.where(y_test == 0)
    bib_rows = np.where(y_test == 2)
    bkg_rows = np.array((qcd_rows[0], bib_rows[0]))
    print(len(bkg_rows[0]))
    prediction = prediction[bkg_rows[0]]
    Z_test = Z_test.iloc[bkg_rows[0]]
    mass_array = (Z_test.groupby(['llp_mH', 'llp_mS']).size().reset_index().rename(columns={0: 'count'}))
    print(mass_array)

    plot_x = []
    plot_y = []
    plot_z = []

    for item, mH, mS in zip(mass_array['count'], mass_array['llp_mH'], mass_array['llp_mS']):
        temp_array = prediction[(Z_test['llp_mH'] == mH) & (Z_test['llp_mS'] == mS)]
        temp_max = np.argmax(temp_array, axis=1)
        temp_num_signal_best = len(temp_max[temp_max == 1])
        temp_eff = temp_num_signal_best / temp_array.shape[0]
        plot_x.append(mH)
        plot_y.append(temp_eff)
        plot_z.append(mS)
        print("mH: " + str(mH) + ", mS: " + str(mS) + ", False positive: " + str(temp_eff))
        f.write("%s,%s,%s\n" % (str(mH), str(mS), str(temp_eff)))

    plt.clf()
    plt.figure()
    plt.scatter(plot_x, plot_y, marker='+', s=150, linewidths=4, c=plot_z, cmap=plt.cm.coolwarm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'mS')
    plt.xlabel("mH")
    plt.ylabel("False Positive Rate")

    plt.savefig(destination + "bkg_falsePositives" + ".pdf", format='pdf', transparent=True)
    plt.clf()


# Make family of ROC Curves
# Since this is a 3-class problem, first make cut on BIB weight for given percentage of correctly tagged BIB
# Take all leftover, and make ROC curve with those
# Make sure to take into account signal and QCD lost in BIB tagged jets
def make_multi_roc_curve(prediction, y, weight, threshold, label, leftovers):
    # Leftover are the indices of jets left after taking out jets below BIB cut
    # So take all those left
    prediction_left = prediction[leftovers]
    y_left = y.values[leftovers]
    weight_left = weight.values[leftovers]

    # Find signal_ratio and qcd_ratio and bib_ratio, ratio of how many signal or qcd or bib left after BIB cut vs how many there were originally
    num_signal_original = y[y == 1].size
    num_signal_leftover = y_left[y_left == 1].size
    signal_ratio = num_signal_leftover / num_signal_original

    num_qcd_original = np.sum(weight.values[y == 0])
    num_qcd_leftover = np.sum(weight_left[y_left == 0])
    qcd_ratio = num_qcd_leftover / num_qcd_original

    num_bib_original = np.sum(weight.values[y == 2])
    num_bib_leftover = np.sum(weight_left[y_left == 2])
    bib_ratio = num_bib_leftover / num_bib_original

    prediction_left_signal = prediction_left[:, 1]

    # If we are looking at BIB cut, then signal vs QCD roc curve
    # Use roc_curve function from scikit-learn
    if label == 2:
        y_roc = label_binarize(y_left, classes=[0, 1, 2])
        (fpr, tpr, _) = roc_curve(y_roc[:, 1], prediction_left_signal, pos_label=1)
        # Scale results by qcd_ratio, signal_ratio
        a = auc((1 - fpr) * qcd_ratio, tpr * signal_ratio)

        # return results of roc curve
        print("FPR:")
        goodIndices = np.where(np.isfinite(1 / fpr))
        print(1 / fpr[goodIndices])
        return (1 / fpr[goodIndices]) * qcd_ratio, tpr[goodIndices] * signal_ratio, a

    # If we are looking at QCD cut, then signal vs BIB roc curve
    # Use roc_curve function from scikit-learn
    if label == 0:
        y_roc = label_binarize(y_left, classes=[0, 1, 2])
        (fpr, tpr, _) = roc_curve(y_roc[:, 1], prediction_left_signal, sample_weight=weight_left, pos_label=1)
        # Scale results by bib_ratio, signal_ratio
        a = auc((1 - fpr) * bib_ratio, tpr * signal_ratio)

        # return results of roc curve
        return (1 / fpr) * bib_ratio, tpr * signal_ratio, a


# Make signal, bib, qcd weight plots
def plot_prediction_histograms(destination,
                               prediction,
                               labels, weight):
    sig_rows = np.where(labels == 1)
    bkg_rows = np.where(labels == 0)
    bib_rows = np.where(labels == 2)
    plt.clf()

    fig, ax = plt.subplots()
    bin_list = np.zeros(1)
    bin_list = np.append(bin_list, np.logspace(np.log10(0.00001), np.log10(1.0), 50))
    ax.hist(prediction[sig_rows][:, 1], color='red', alpha=0.5, linewidth=0, histtype='stepfilled', bins=bin_list,
            label="Signal")
    ax.hist(prediction[bkg_rows][:, 1], color='blue', alpha=0.5, linewidth=0, histtype='stepfilled', bins=bin_list,
            label="QCD")
    ax.hist(prediction[bib_rows][:, 1], color='green', alpha=0.5, linewidth=0, histtype='stepfilled', bins=bin_list,
            label="BIB")
    plt.yscale('log', nonposy='clip')
    plt.xscale('log', nonposx='clip')
    ax.set_xlabel("Signal NN weight")
    ax.legend(loc='best')

    plt.savefig(destination + "sig_predictions" + ".pdf", format='pdf', transparent=True)
    plt.clf()

    fig, ax = plt.subplots()
    ax.hist(prediction[sig_rows][:, 0], weights=weight.values[sig_rows], color='red', alpha=0.5, linewidth=0,
            histtype='stepfilled', bins=bin_list, label="Signal")
    ax.hist(prediction[bkg_rows][:, 0], weights=weight.values[bkg_rows], color='blue', alpha=0.5, linewidth=0,
            histtype='stepfilled', bins=bin_list, label="QCD")
    ax.hist(prediction[bib_rows][:, 0], weights=weight.values[bib_rows], color='green', alpha=0.5, linewidth=0,
            histtype='stepfilled', bins=bin_list, label="BIB")
    plt.yscale('log', nonposy='clip')
    ax.set_xlabel("QCD NN weight")
    plt.xscale('log', nonposx='clip')
    ax.legend()

    plt.savefig(destination + "qcd_predictions" + ".pdf", format='pdf', transparent=True)
    plt.clf()

    fig, ax = plt.subplots()
    ax.hist(prediction[sig_rows][:, 2], weights=weight.values[sig_rows], color='red', alpha=0.5, linewidth=0,
            histtype='stepfilled', bins=bin_list, label="Signal")
    ax.hist(prediction[bkg_rows][:, 2], weights=weight.values[bkg_rows], color='blue', alpha=0.5, linewidth=0,
            histtype='stepfilled', bins=bin_list, label="QCD")
    ax.hist(prediction[bib_rows][:, 2], weights=weight.values[bib_rows], color='green', alpha=0.5, linewidth=0,
            histtype='stepfilled', bins=bin_list, label="BIB")
    plt.yscale('log', nonposy='clip')
    ax.set_xlabel("BIB NN weight")
    plt.xscale('log', nonposx='clip')
    ax.legend()

    plt.savefig(destination + "bib_predictions" + ".pdf", format='pdf', transparent=True)
    plt.clf()
    return
