import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_events_hfo(data):
    
    channels = data.unique()
    counts = np.array([])
    
    for channel in channels:
        counts = np.append(
            counts,
            len(data[data == channel])
        )

    plt.style.use('seaborn-v0_8-darkgrid')

    plt.bar(channels, counts, color = '#ff8500', width = 0.4)
    plt.title('iHFO')
    plt.xlabel('Channels')    
    plt.ylabel('Counts')
    plt.xticks(rotation = 90, fontsize = 10)

    # plt.savefig('prueba_chart.png', dpi=300, bbox_inches='tight')

    plt.show()


    

