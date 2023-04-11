import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.filters import threshold_otsu

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
    percentil = np.percentile(df['hfo_rate'], 75)
    ax.axhline(y = percentil, color='r', linestyle='--')
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    # handles = [plt.Rectangle((0,0),1,1, color = df['colors']) for label in labels]
    ax.legend(handles, labels)
    
    plt.title('iHFO')
    plt.xlabel('Channels')    
    plt.ylabel('RFreq [Counts/min]')
    plt.xticks(rotation = 90, fontsize = 10)
    plt.show()   

def histogram_hfo(df):
    
    hist, bin_edges = np.histogram(df['hfo_rate'], bins=10)
    bin_edges = bin_edges[0:len(hist)]

    plt.hist(df['hfo_rate'], bins=10, alpha=0.4, color='blue')
    plt.axvline(threshold_otsu(hist = [hist, bin_edges]), color='red')
    plt.show()

    # Calcular el percentil 90 de los datos
    percentil_90 = np.percentile(df['hfo_rate'], 85)

    plt.hist(df['hfo_rate'], bins=10, alpha=0.4, color='blue')
    plt.axvline(percentil_90, color='red')
    plt.show()
