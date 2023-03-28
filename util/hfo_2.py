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
    


    

