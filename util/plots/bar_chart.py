import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bar_chart(df):          

    status = df['status'].unique() 
    colors = {}
    for state, color in zip(status, df['colors'].unique()):
        colors[state] = color
    labels = list(colors.keys())        
   
    #bar_chart
    fig, ax = plt.subplots()
    # ax.bar(df['channels'], df['counts']/df['counts'].max(), color=df['colors'], width=0.4)
    ax.bar(df['channels'], df['hfo_rate'], color=df['colors'], width=0.4)
    # ax.axhline(y=0.75, color='r', linestyle='--')
    percentil_90 = np.percentile(df['hfo_rate'], 90)
    ax.axhline(y = percentil_90, color='r', linestyle='--')
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    # handles = [plt.Rectangle((0,0),1,1, color = df['colors']) for label in labels]
    ax.legend(handles, labels)
    
    plt.title('iHFO')
    plt.xlabel('Channels')    
    plt.ylabel('RFreq [Counts/min]')
    plt.xticks(rotation = 90, fontsize = 10)
    plt.show()   