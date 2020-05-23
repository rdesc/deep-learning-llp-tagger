import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def pt_ordering_plot(data, pt_ordering=""):
    """
    Plots jets to illustrate pt ordering

    @param data: pandas dataframe
    @param pt_ordering: ordering of pt (default is descending pt order)
    """
    # matplotlib config
    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 160
    mpl.rcParams['savefig.dpi'] = 200
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['figure.titlesize'] = 'medium'

    # get clus pt columns
    clus_pt_cols = [col for col in data if col.startswith('clus_pt')]
    # get track pt columns
    track_pt_cols = [col for col in data if col.startswith('nn_track_pt')]
    # clus x-axis
    x_clus = np.arange(0, len(clus_pt_cols))
    # track x-axis
    x_track = np.arange(0, len(track_pt_cols))

    # get average for each column
    clus_data = data[clus_pt_cols].mean(axis=0)
    track_data = data[track_pt_cols][data[track_pt_cols] < 10].mean(axis=0)  # ignore pt > 10

    # initialize plot
    fig, ax_clus = plt.subplots()

    ax_clus.set_xlabel('column #')
    ax_clus.set_ylabel('transverse momentum (pT)')
    line1, = ax_clus.plot(x_clus, clus_data, '-', color='red', label='cluster')
    ax_clus.tick_params(axis='y', labelcolor='red')

    ax_track = ax_clus.twinx()
    line2, = ax_track.plot(x_track, track_data, '-', color='blue', label='track')
    ax_track.tick_params(axis='y', labelcolor='blue')

    plt.legend((line1, line2), ('cluster', 'track'))
    ax_track.set_title('pT Ordering: descending' if not pt_ordering else 'pT Ordering: ' + pt_ordering)

    plt.savefig('pt_ordering_descending.png' if not pt_ordering else 'pt_ordering_' + pt_ordering + '.png')
