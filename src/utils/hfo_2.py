import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_events_hfo_2(data, ch_info, last_time):
    
    channels = data.unique()
    counts = np.array([])

    ch_names_last = []
    for channel in channels:
        counts = np.append(
            counts,
            len(data[data == channel])
        )
        ch_names_last.append( channel.split('-')[0] )          
    
    status_desc = []
    for channel in ch_names_last:
        status_desc.append(ch_info.loc[ch_info['name'] == channel, 'status_description'].iloc[0])

    hfo_rate = counts / last_time * 60 #frecuencia de eventos [counts]/min
    df = pd.DataFrame({'channels': channels,
                    'ch_split': ch_names_last,
                    'counts': counts,
                    'status': status_desc,   
                    'hfo_rate': hfo_rate.round(), #redondeo de la frec
                    })

    status = df['status'].unique()
    #define colors 
    colors = []
    for i in range(df.shape[0]): colors.append('')
    df =df.assign(colors = colors) # or df['colors'] = colors
    color = 0x4e4e4e               #0x para notación hexadecimal de RGB
    for state in status:    
        for i in range(df.shape[0]):
            if (str(df.at[i, 'status']).casefold() == str(state).casefold()):
                df.at[i, 'colors'] ='#' + hex(color)[2:]
        color = color + 0x006456        
        
    return df        
    
def algorithm_params(lower_freq, upper_freq, band_width, window_min, window_max, window_step):
    """
    objective: create array with differents algohortim configuration.
    The objective of the function is to generate an array with different algorithm configurations for the detection of high frequency oscillations (HFOs) in EEG data. The function takes as input several parameters related to the frequency bands and window sizes to be used in the HFO detection algorithm.
    
    Output: 
    - all_comb: an array containing tuples representing all the different algorithm configurations to be tested in the HFO detection process.

    """
    band_width_2 = upper_freq - lower_freq
    filter_bands = []
    filter_bands.append((lower_freq, upper_freq))

    for i in range(round(band_width_2 / (band_width / 2)) - 1):
        if lower_freq + 30 * i + band_width == upper_freq + 1:
            band_width = band_width - 1
        filter_bands.append(tuple([lower_freq + 30 * i, lower_freq + 30 * i + band_width]))   

    all_combinations = product(filter_bands, range(window_min, window_max + 1, window_step))

    return all_combinations
        
    # end def
# end def

    

