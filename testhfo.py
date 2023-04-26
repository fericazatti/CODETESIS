import mne
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from mne_connectivity import TemporalConnectivity

from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle

from mne_hfo import LineLengthDetector, RMSDetector
from mne_hfo.score import _compute_score_data, accuracy
from mne_hfo.sklearn import make_Xy_sklearn, DisabledCV

# from epycom.event_detection import HilbertDetector

from util.general_utils import convert_to_bipolar
from util.hfo import plot_events_hfo
from util.hfo_2 import plot_events_hfo_2
from util.plots.hfo_plots import *

from skimage.filters import threshold_otsu
import pandas as pd
import xarray as xr

from read_two_bids import raws, cahnnels_info_paths
from itertools import product


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

#########import data
raw = mne.io.read_raw_edf('ds004100/sub-HUP060/ses-presurgery/ieeg/sub-HUP060_ses-presurgery_task-interictal_acq-seeg_run-01_ieeg.edf', preload = True) 
ch_info = pd.read_csv('ds004100/sub-HUP060/ses-presurgery/ieeg/sub-HUP060_ses-presurgery_task-interictal_acq-seeg_run-01_channels.tsv', sep = '\t')

# %% Query bads and surgical channels
ch_bads = ch_info.loc[ch_info['status'] == 'bad']['name']
ch_surgical = ch_info.loc[ch_info['status_description'] == 'soz']['name']

# %% delete from raw file bad channels
raw.drop_channels(ch_bads)

# %% comvert to bipolar reference
raw = convert_to_bipolar(raw)

# for i in range(len(raws)):
#     raws[i] = convert_to_bipolar(raws[i])

# Set Key Word Arguments for the Line Length Detector and generate the class object
kwargs = {
    'filter_band': (80, 120), # (l_freq, h_freq)
    'threshold': 3, # Number of st. deviations
    'win_size': 100, # Sliding window size in samples
    'overlap': 0.25, # Fraction of window overlap [0, 1]
    'hfo_name': "ripple"
}

rms_detector = RMSDetector(**kwargs)

#### Set
ll_detector = LineLengthDetector(**kwargs)


# ll_detector = ll_detector.fit(raw)

####### Pandas dataframe containing onset, duration, sample trial, and trial type per HFO
# ll_hfo_df = ll_detector.df_

# Detect HFOs in the raw data using the RMSDetector method.

rms_detector = rms_detector.fit(raw)

### Outpout detector to dataframe pandas
rms_hfo_df = rms_detector.df_

# plot_events_hfo(ll_hfo_df['channels']) #la funcion recibe la serie de canales en los eventos encontrados
# plot_events_hfo(rms_hfo_df['channels'])

# plot_events_hfo_2(ll_hfo_df['channels'], ch_info)
hfo_dist_df = plot_events_hfo_2(rms_hfo_df['channels'], ch_info, raw._last_time)

###### Generación del histograma y posterior umbralización
bar_chart(hfo_dist_df)

histogram_hfo(hfo_dist_df)

for raw in raws:
    rms_detector = rms_detector.fit(raw)

    ### Outpout detector to dataframe pandas
    rms_hfo_df = rms_detector.df_

    # plot_events_hfo(ll_hfo_df['channels']) #la funcion recibe la serie de canales en los eventos encontrados
    plot_events_hfo(rms_hfo_df['channels'])

    # plot_events_hfo_2(ll_hfo_df['channels'], ch_info)
    # hfo_dist_df = plot_events_hfo_2(rms_hfo_df['channels'], ch_info, raw._last_time)

    ###### Generación del histograma y posterior umbralización
    # bar_chart(hfo_dist_df)

    # histogram_hfo(hfo_dist_df)

# %% 
kwargs = {
    'lower_freq':   40,
    'upper_freq':   249,
    'band_width':   60,
    'window_min':   50,
    'window_max':   150,
    'window_step':  20
}

# definir las dimensiones y sus etiquetas
# algorithms_config = []
# channels = ['channel1', 'channel2', 'channel3', 'channel4']
# values_coords = ['counts rate', 'status', 'color']

algorithms_params_names = []
algorithms_params_array = []
status = ['bad' if element in raw.info['bads'] else 'good' for element in raw.info['ch_names']] 
# %%
for param_combine in algorithm_params(**kwargs):
     
    kwargs = {
        'filter_band': param_combine[0], # (l_freq, h_freq)
        'threshold': 3, # Number of st. deviationsG
        'win_size': param_combine[1], # Sliding window size in samples
        'overlap': 0.25, # Fraction of window overlap [0, 1]
        'hfo_name': "ripple"
    }
    
    rms_detector = RMSDetector(**kwargs) 
    rms_detector = rms_detector.fit(raw)
    
    rms_hfo_df = rms_detector.df_
    hfo_dist_df = plot_events_hfo_2(rms_hfo_df['channels'], ch_info, raw._last_time)
    hfo_dist_df['status_acquis'] = status
    
    hfo_dist_xarray = (hfo_dist_df.set_index(['channels']).to_xarray()).to_array()
    hfo_dist_xarray = hfo_dist_xarray.rename({'variable':'values'})    
    
    algorithms_params_array.append(hfo_dist_xarray)
    algorithms_params_names.append(f'bw-{param_combine[0]}_ww-{param_combine[1]}')
    

subject_dataset = xr.Dataset(
    dict(
        zip(algorithms_params_names, algorithms_params_array)
        )
    )

attributes = raw.info['subject_info']
subject_dataset = subject_dataset.assign_attrs(**attributes)


# %%
