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

from util.general_utils import convert_to_bipolar
from util.hfo import plot_events_hfo

import pandas as pd

# matplotlib.use('pdf')

#import data
raw = mne.io.read_raw_edf('subjects/sub-HUP139/sub-HUP139_ses-presurgery_ieeg_sub-HUP139_ses-presurgery_task-interictal_acq-seeg_run-02_ieeg.edf', preload = True) 

# raw = mne.io.read_raw_edf('/home/proyectoepilepsia/Documents/Pruebas/CodeTesis/subjects/sub-HUP139/Reduced_sub-HUP139_ses-presurgery_task-interictal_acq-seeg_run-02_ieeg.edf',
#                            preload=True)

# raw = mne.io.read_raw_edf('/home/proyectoepilepsia/Documents/Pruebas/CodeTesis/subjects/sub-HUP139/sub-HUP139_ses-presurgery_task-interictal_acq-seeg_run-01_ieeg.edf',
#                            preload=True)
ch_info = pd.read_csv('subjects/sub-HUP139/sub-HUP139_ses-presurgery_ieeg_sub-HUP139_ses-presurgery_task-interictal_acq-seeg_run-01_channels.tsv', sep = '\t')
ch_localization = pd.read_csv('subjects/sub-HUP139/sub-HUP139_ses-presurgery_ieeg_sub-HUP139_ses-presurgery_acq-seeg_space-fsaverage_electrodes.tsv', sep = '\t')

ch_bads = ch_info.loc[ch_info['status'] == 'bad']['name']
ch_surgical = ch_info.loc[ch_info['status_description'] == 'soz']['name']

raw.drop_channels(ch_bads)

raw = convert_to_bipolar(raw)

# ch_info = pd.read_csv('subjects/sub-HUP139/sub-HUP139_ses-presurgery_ieeg_sub-HUP139_ses-presurgery_task-interictal_acq-seeg_run-01_channels.tsv', sep = '\t')
# ch_localization = pd.read_csv('subjects/sub-HUP139/sub-HUP139_ses-presurgery_ieeg_sub-HUP139_ses-presurgery_acq-seeg_space-fsaverage_electrodes.tsv', sep = '\t')

# ch_bads = ch_info.loc[ch_info['status'] == 'bad']['name']



#read annotations from file
ann = raw.annotations

# Set Key Word Arguments for the Line Length Detector and generate the class object
kwargs = {
    'filter_band': (80, 249), # (l_freq, h_freq)
    'threshold': 3, # Number of st. deviations
    'win_size': 100, # Sliding window size in samples
    'overlap': 0.25, # Fraction of window overlap [0, 1]
    'hfo_name': "ripple"
}

# Set Key Word Arguments for the RMS Detector and generate the class object
kwargs = {
    'filter_band': (80, 500),
    'threshold': 3,
    'win_size': 100,
    'overlap': 0.25,
    'hfo_name': 'ripple',
}
rms_detector = RMSDetector(**kwargs)

ll_detector = LineLengthDetector(**kwargs)

ll_detector = ll_detector.fit(raw)

# Dictionary where keys are channel index and values are a list of tuples in the form of (start_samp, end_samp)
ll_chs_hfo_dict = ll_detector.chs_hfos_
# nCh x nWin ndarray where each value is the line-length of the data window per channel
ll_hfo_event_array = ll_detector.hfo_event_arr_
# Pandas dataframe containing onset, duration, sample trial, and trial type per HFO
ll_hfo_df = ll_detector.df_

# Detect HFOs in the raw data using the RMSDetector method.
rms_detector = rms_detector.fit(raw)

rms_chs_hfo_dict = rms_detector.chs_hfos_
rms_hfo_event_array = rms_detector.hfo_event_arr_
rms_hfo_df = rms_detector.df_

plot_events_hfo(ll_hfo_df['channels']) #la funcion recibe la serie de canales en los eventos encontrados
plot_events_hfo(rms_hfo_df['channels'])

# list_hfo = ll_chs_hfo_dict['RAFa1-RAFa2']

# sfreq = raw.info['sfreq']

# for sample_par in list_hfo:
#     onset = sample_par[0] * 1/sfreq    
#     duration = sample_par[1] * 1/sfreq - onset
#     ann.append(onset, duration, 'EventHfo')

# #create new annotations
# ann.append(
#     [2.5 , 25., 50., 107.5, 120.5],  #onset in seconds
#     [2.5, 2.5, 2.5, 2.5, 2.5],     #duration 
#     ['test1', 'test1', 'test1', 'test1', 'test1']) #description

# #graphics of raw data and annotations
# raw.plot(highpass=0.1, scalings=dict(eeg=40e-6)) 


# raw.plot(  )
# raw.plot_psd(  )

# event_id = {
#     #'test1': 1, 
#     'EventHfo': 2
#     }

# #converting annotations objects to event array
# events,_ =mne.events_from_annotations(
#     raw,
#     event_id=event_id)
#     # chunk_duration=30.)

# # plot events
# fig =mne.viz.plot_events(events, event_id=event_id,
#                           sfreq=raw.info['sfreq'],
#                           first_samp=events[0, 0])

# #Events can also be plotted alongside the Raw object  
# raw.plot(events=events, start=5, duration=100, color='gray',
#          event_color={2: 'r'})

# epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.9)
    
# epochs.plot()


# fmin = 8.
# fmax = 13.
# sfreq = raw.info['sfreq']  # the sampling frequency
# con_methods = ['pli', 'coh', 'ciplv']
# # con_methods = ['pli']
# conn = []
# for method in con_methods:
#     conn.append(
#         spectral_connectivity_epochs(
#             data = epochs, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin,
#             fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1
#         )
#     )

# conn = np.array(conn)           

# figure, axes = plt.subplots(1, 3, figsize=(8, 4), facecolor='black',
#                          subplot_kw=dict(polar=True))

# for ax, x in zip(axes, conn):  
#     plot_connectivity_circle(x.get_data(output='dense')[:,:,0],
#                             n_lines = 100,
#                             node_names = raw.ch_names,
#                             title='All-to-All Connectivity ' + x.method[0],
#                             padding=0, fontsize_colorbar=6,
#                             ax = ax)           

# figure.show() # to see complete subplot

# ###### grafica de una sola matriz de conectividad a la vez
# # for x in conn:
# #     plot_connectivity_circle(x.get_data(output='dense')[:,:,0],
# #                             node_names = raw.ch_names,
# #                             title='All-to-All Connectivity ' + x.method[0])           

