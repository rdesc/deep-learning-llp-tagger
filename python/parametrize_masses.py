import seaborn as sns
import numpy as np
import pandas as pd

def parametrize_masses(data):

    mass_array = (data.groupby(['llp_mH','llp_mS']).size().reset_index().rename(columns={0:'count'}))
    mass_array = mass_array[mass_array['llp_mH'] != 0]
    mass_array['proportion'] = mass_array['count']/len(data.loc[data['label'] == 1].index)

    non_signal_length = len(data.loc[ data['label'] != 1].index)
    data = data.reset_index()


    index_sum=0
    counter=0

    non_signal_data = data.loc[ data['label'] != 1]
    initial_index = non_signal_data.index[0]

    for item,mH,mS in zip(mass_array['proportion'],mass_array['llp_mH'],mass_array['llp_mS']):
        end_index = initial_index+index_sum+round(item*non_signal_length)
        counter=counter+1
        if counter == len(mass_array.index):
            end_index = initial_index + non_signal_length
        non_signal_data.loc[initial_index+index_sum:end_index,'llp_mH'] = mH
        non_signal_data.loc[initial_index+index_sum:end_index,'llp_mS'] = mS
        if counter == len(mass_array.index):
            index_sum = end_index
        else: 
            index_sum = index_sum+ round(item*non_signal_length)
            

    print("OG length: " + str(non_signal_length) )
    print("New length: " + str(data.loc[ data['label'] != 1].shape[0]) )

    if non_signal_length != data.loc[ data['label'] != 1].shape[0]:
        print("LENGTHS NOT EQUAL")
        return 1

    data.loc[ data['label'] != 1] = non_signal_data


  
    return data

