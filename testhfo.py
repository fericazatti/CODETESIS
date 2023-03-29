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

#########import data

# raw = mne.io.read_raw_edf('subjects/alemanno/ALEMANO_34_FEDERICO.edf',preload = True)
#raw = mne.io.read_raw_edf('subjects/sub-HUP160/sub-HUP160_ses-presurgery_ieeg_sub-HUP160_ses-presurgery_task-interictal_acq-seeg_run-01_ieeg.edf', preload = True) 
raw = mne.io.read_raw_edf('/home/proyectoepilepsia/Documents/software/CODETESIS/subjects/sub-HUP060/sub-HUP060_ses-presurgery_task-interictal_acq-seeg_run-02_ieeg.edf',
                            preload=True)
# raw = mne.io.read_raw_edf('/home/proyectoepilepsia/Documents/Pruebas/CodeTesis/subjects/sub-HUP160/sub-HUP160_ses-presurgery_task-interictal_acq-ecog_run-01_ieeg.edf',
#                            preload=True)
ch_info = pd.read_csv('subjects/sub-HUP060/sub-HUP060_ses-presurgery_task-interictal_acq-seeg_run-02_channels.tsv', sep = '\t')
# ch_localization = pd.read_csv('subjects/sub-HUP160/sub-HUP160_ses-presurgery_ieeg_sub-HUP160_ses-presurgery_acq-ecog_space-fsaverage_electrodes.tsv', sep = '\t')

### Query bads and surgical channels
ch_bads = ch_info.loc[ch_info['status'] == 'bad']['name']
ch_surgical = ch_info.loc[ch_info['status_description'] == 'soz']['name']

#### delete from raw file bad channels
raw.drop_channels(ch_bads)

raw = convert_to_bipolar(raw)

# Set Key Word Arguments for the Line Length Detector and generate the class object
kwargs = {
    'filter_band': (80, 249), # (l_freq, h_freq)
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