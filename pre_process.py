import numpy as np


def pre_process(data, min_pt, max_pt, pt_ordering=""):
    # Now For Muon Segments
    print("Pre-processing Muon Segments")

    # Get all eta Position columns
    filter_MSeg_eta = [col for col in data if col.startswith("nn_MSeg_etaPos")]
    # Get all phi Position columns
    filter_MSeg_phi = [col for col in data if col.startswith("nn_MSeg_phiPos")]
    # Get all phi Direction  columns
    filter_MSeg_phiDir = [col for col in data if col.startswith("nn_MSeg_phiDir")]

    # Subtract the eta of the jet from all MSegs
    data[filter_MSeg_eta] = data[filter_MSeg_eta].sub(data["jet_eta"], axis='index')

    # Subtract the phi of the jet from all MSegs
    data[filter_MSeg_phi] = data[filter_MSeg_phi].sub(data["jet_phi"], axis='index')

    # Subtract the phi of the jet from all MSegs Dir
    data[filter_MSeg_phiDir] = data[filter_MSeg_phiDir].sub(data["jet_phi"], axis='index')

    print(list(data[filter_MSeg_phi].iloc[1, :]))
    print(list(data[filter_MSeg_eta].iloc[1, :]))

    np.set_printoptions(precision=17)

    print("Pre-processing jets")

    print(max_pt)
    print(min_pt)

    # SCALE JET PT
    data["jet_pt"] = data["jet_pt"].sub(float(min_pt), axis='index')
    data["jet_pt"] = data["jet_pt"].divide((float(max_pt) - float(min_pt)), axis='index')

    print("Pre-processing clusters")
    # DO PHI, ETA Shift

    # Get all eta columns
    filter_clus_eta = [col for col in data if col.startswith("clus_eta")]
    # Get all phi columns
    filter_clus_phi = [col for col in data if col.startswith("clus_phi")]
    # Get all pt columns
    filter_clus_pt = [col for col in data if col.startswith("clus_pt")]

    # Subtract the eta of first cluster(largest pt) from all other
    data[filter_clus_eta] = data[filter_clus_eta].sub(data["clus_eta_0"], axis='index')

    # Subtract the phi of first cluster(largest pt) from all other
    data[filter_clus_phi] = data[filter_clus_phi].sub(data["clus_phi_0"], axis='index')

    # Do eta, phi FLIP

    # Add all etas weighted by pt, then make column that is 1 if positive, -1 if negative
    data['clus_sign'] = np.sum(
        np.multiply(data[filter_clus_eta].fillna(0).to_numpy(), data[filter_clus_pt].fillna(0).to_numpy()), axis=1)
    data['clus_sign'] = data['clus_sign'].apply(lambda x: 1 * (x >= 0) + (-1) * (x < 0))

    # Flip (multiply by -1) according to previously calculated column
    data[filter_clus_eta] = data[filter_clus_eta].multiply(data["clus_sign"], axis='index')

    # SCALE CLUSTER PT
    data[filter_clus_pt] = data[filter_clus_pt].sub(min_pt, axis='index')
    data[filter_clus_pt] = data[filter_clus_pt].divide((max_pt - min_pt), axis='index')

    print("Pre-processing cluster energy fraction")

    # SCALE Cluster Energy Fraction, then unites layers across different eta ranges
    for i in range(0, 30):
        filter_clus_eFrac = [col for col in data if col.startswith("clus_l") and col.endswith('_' + str(i))]
        data['sum_eFrac'] = data[filter_clus_eFrac].sum(axis=1)

        data['clus_l1ecal_' + str(i)] = data['clus_l1ecal_' + str(i)].divide(data['sum_eFrac'], axis='index')
        data['clus_l2ecal_' + str(i)] = data['clus_l2ecal_' + str(i)].divide(data['sum_eFrac'], axis='index')
        data['clus_l3ecal_' + str(i)] = data['clus_l3ecal_' + str(i)].divide(data['sum_eFrac'], axis='index')
        data['clus_l4ecal_' + str(i)] = data['clus_l4ecal_' + str(i)].divide(data['sum_eFrac'], axis='index')

        data['clus_l1hcal_' + str(i)] = data['clus_l1hcal_' + str(i)].divide(data['sum_eFrac'], axis='index')
        data['clus_l2hcal_' + str(i)] = data['clus_l2hcal_' + str(i)].divide(data['sum_eFrac'], axis='index')
        data['clus_l3hcal_' + str(i)] = data['clus_l3hcal_' + str(i)].divide(data['sum_eFrac'], axis='index')
        data['clus_l4hcal_' + str(i)] = data['clus_l4hcal_' + str(i)].divide(data['sum_eFrac'], axis='index')

    # Delete calculation variable
    del data['sum_eFrac']

    # Now For Tracks
    print("Pre-processing tracks")

    # Get all eta columns
    filter_track_eta = [col for col in data if col.startswith("nn_track_eta")]
    # Get all phi columns
    filter_track_phi = [col for col in data if col.startswith("nn_track_phi")]
    # Get all pt columns
    filter_track_pt = [col for col in data if col.startswith("nn_track_pt")]
    # Get all z vertex columns
    filter_track_vertex_z = [col for col in data if col.startswith("nn_track_vertex_z")]

    # Subtract the eta of the jet from all tracks
    data[filter_track_eta] = data[filter_track_eta].sub(data["jet_eta"], axis='index')

    # Subtract the phi of the jet from all tracks
    data[filter_track_phi] = data[filter_track_phi].sub(data["jet_phi"], axis='index')

    # Do eta, phi FLIP

    # SCALE Track PT
    data[filter_track_pt] = data[filter_track_pt].sub(min_pt * 1000, axis='index')
    data[filter_track_pt] = data[filter_track_pt].divide((max_pt * 1000 - min_pt * 1000), axis='index')

    data[filter_track_vertex_z] = data[filter_track_vertex_z].divide((100), axis='index')

    # SCALE Track z0
    filter_track_z0 = [col for col in data if col.startswith("nn_track_z0")]
    data[filter_track_z0] = data[filter_track_z0].divide(250, axis='index')

    # Add all etas weighted by pt, then make column that is 1 if positive, -1 if negative
    data['track_sign'] = np.sum(
        np.multiply(data[filter_track_eta].fillna(0).to_numpy(), data[filter_track_pt].fillna(0).to_numpy()), axis=1)
    data['track_sign'] = data['track_sign'].apply(lambda x: 1 * (x >= 0) + (-1) * (x < 0))
    # Flip (multiply by -1) according to previously calculated column
    data[filter_track_eta] = data[filter_track_eta].multiply(data["track_sign"], axis='index')

    # Do pt re-ordering if specified by pt_ordering arg
    if pt_ordering != "descending" and pt_ordering:

        # NOTE: all these columns in dataframe will change from dtype float32 to float64 -> increases size of df
        clusters_col_names = ['clus_pt', 'clus_eta', 'clus_phi', 'clus_l1hcal', 'clus_l1ecal',
                              'clus_l2hcal', 'clus_l2ecal', 'clus_l3hcal', 'clus_l3ecal', 'clus_l4ecal',
                              'clus_l4hcal', 'clus_time']
        tracks_col_names = ['nn_track_pt', 'nn_track_eta', 'nn_track_phi', 'nn_track_vertex_nParticles',
                            'nn_track_vertex_x', 'nn_track_vertex_y', 'nn_track_vertex_z', 'nn_track_d0',
                            'nn_track_z0', 'nn_track_chiSquared', 'nn_track_PixelShared',
                            'nn_track_SCTShared', 'nn_track_PixelHoles', 'nn_track_SCTHoles',
                            'nn_track_PixelHits', 'nn_track_SCTHits']

        if pt_ordering == "ascending":
            # Re-order jet clusters by ascending pt
            print("Ordering jet clusters by ascending pt...")
            np_clus_pt = np.absolute(data[filter_clus_pt].to_numpy())
            np_clus_pt = np.around(np_clus_pt, decimals=6)
            clus_index_array = np.argsort(np_clus_pt, axis=1, kind='mergesort')
            for item in clusters_col_names:
                filter_temp = [col for col in data if col.startswith(item)]
                data_temp = data[filter_temp].to_numpy()
                data_temp = np.array(list(map(lambda x, y: y[x], clus_index_array, data_temp)))
                data[filter_temp] = data_temp

            # Re-order tracks by ascending pt
            print("Ordering tracks by ascending pt...")
            np_track_pt = np.absolute(data[filter_track_pt].to_numpy())
            np_track_pt = np.around(np_track_pt, decimals=6)
            track_index_array = np.argsort(np_track_pt, axis=1, kind='mergesort')
            tracks_col_names = ['nn_track_eta', 'nn_track_phi', 'nn_track_vertex_z', 'nn_track_z0', 'nn_track_pt']
            for item in tracks_col_names:
                filter_temp = [col for col in data if col.startswith(item)]
                data_temp = data[filter_temp].to_numpy()
                data_temp = np.array(list(map(lambda x, y: y[x], track_index_array, data_temp)))
                data[filter_temp] = data_temp

        elif pt_ordering == "random":
            # Re-order jet clusters in a random pt order
            print("Ordering jet clusters by random pt order...")
            np_clus_pt = np.absolute(data[filter_clus_pt].to_numpy())
            num_cols = np_clus_pt[0].size
            nans = np.isnan(np_clus_pt)
            clus_index_array = np.full((np.shape(np_clus_pt)[0], num_cols), 1)
            size = clus_index_array.shape[0]
            for i in range(size):
                nan_array = np.where(nans[i] == True)[0]
                first_nan_index = (
                    nan_array[0] if nan_array.size > 0 else num_cols - 1)  # handles case when there are no nans
                temp_array = np.arange(0, first_nan_index)
                np.random.shuffle(temp_array)
                temp_array = np.concatenate((temp_array, np.arange(first_nan_index, num_cols)))
                clus_index_array[i] = temp_array

            for item in clusters_col_names:
                filter_temp = [col for col in data if col.startswith(item)]
                data_temp = data[filter_temp].to_numpy()
                data_temp = np.array(list(map(lambda x, y: y[x], clus_index_array, data_temp)))
                data[filter_temp] = data_temp

            # Re-order tracks in a random pt order
            print("Ordering tracks by random pt order...")
            np_track_pt = np.absolute(data[filter_track_pt].to_numpy())
            num_cols = np_track_pt[0].size
            nans = np.isnan(np_track_pt)
            track_index_array = np.full((np.shape(np_track_pt)[0], num_cols), 1)
            size = track_index_array.shape[0]
            for i in range(size):
                nan_array = np.where(nans[i] == True)[0]
                first_nan_index = (nan_array[0] if nan_array.size > 0 else num_cols - 1)
                temp_array = np.arange(0, first_nan_index)
                np.random.shuffle(temp_array)
                temp_array = np.concatenate((temp_array, np.arange(first_nan_index, num_cols)))
                track_index_array[i] = temp_array

            for item in tracks_col_names:
                filter_temp = [col for col in data if col.startswith(item)]
                data_temp = data[filter_temp].to_numpy()
                data_temp = np.array(list(map(lambda x, y: y[x], track_index_array, data_temp)))
                data[filter_temp] = data_temp
        else:
            print("Ordering " + pt_ordering + " not recognized, ignoring...")

    # Shuffle all jets
    print("Shuffling pre-processed data")
    data = data.sample(frac=1).reset_index(drop=True)

    return data
