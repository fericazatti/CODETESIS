import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_events_hfo_2(data, ch_info):
    
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


    df = pd.DataFrame({'channels': channels, 'ch_split': ch_names_last, 'counts': counts, 'status': status_desc})
    status = df['status'].unique()

    colors = []
    for i in range(df.shape[0]): colors.append('')
    df =df.assign(colors= colors)
    color = 0x4e4e4e
    for state in status:    
        for i in range(df.shape[0]):
            if (str(df.at[i, 'status']).casefold() == str(state).casefold()):
                df.at[i, 'colors'] ='#' + hex(color)[2:]
        color = color + 0x006456
        
    colors = {}
    for state, color in zip(status, df['colors'].unique()):
        colors[state] = color
    labels = list(colors.keys())
    fig, ax = plt.subplots()
    ax.bar(df['channels'], df['counts']/df['counts'].max(), color=df['colors'], width=0.4)
    ax.axhline(y=0.75, color='r', linestyle='--')
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    ax.legend(handles, labels)
    plt.title('iHFO')
    plt.xlabel('Channels')    
    plt.ylabel('Counts')
    plt.xticks(rotation = 90, fontsize = 10)
    plt.show()
        
    


    

