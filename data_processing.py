# %% Imports
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

from src.utils.preprocessing import convert_to_bipolar
from src.utils.hfo import plot_events_hfo
from src.utils.hfo_2 import plot_events_hfo_2, algorithm_params
from src.utils.plots.hfo_plots import *

from skimage.filters import threshold_otsu
import pandas as pd
import xarray as xr

# from read_two_bids import raws, cahnnels_info_paths
from itertools import product

from src.utils.read_data_from_bids import detect_subjects, read_four_subjects

import pickle

    # %% 
def data_processing(kwargs_bids, kwargs_for_combine):
    
    # id query of the patients in BIDS that meet the kwargs_bids
    subjects_id = detect_subjects(**kwargs_bids)

    # Calculamos el número de filas necesarias
    rows = len(subjects_id) // 4 + (1 if len(subjects_id) % 4 != 0 else 0)

    # Creamos la matriz bidimensional con valores None
    subjects_matrix = [[None] * 4 for _ in range(rows)]

    # Llenamos la matriz con los elementos del arreglo
    for i, element in enumerate(subjects_id):
        row = i // 4
        column = i % 4
        subjects_matrix[row][column] = element

    # %% 
    subject_datasets = []

    for subjects_row in subjects_matrix:
        #read raw data on bids
        raws, channels = read_four_subjects(subjects = subjects_row, **kwargs_bids)
        
        #preprocessing raw data -> convert to bipolar montage
        for i in enumerate(raws):
            raws[i[0]] = convert_to_bipolar(raws[i[0]])
        
        for raw, channel_info_path in zip(raws, channels):
            ch_info = pd.read_csv(channel_info_path, sep='\t')
            
            algorithms_params_names = []
            algorithms_params_array = []

            for combined_parameters in algorithm_params(**kwargs_for_combine):
                
                kwargs_rmsdetector = {
                    'filter_band': combined_parameters[0], # (l_freq, h_freq)
                    'threshold': 3, # Number of st. deviationsG
                    'win_size': combined_parameters[1], # Sliding window size in samples
                    'overlap': 0.25, # Fraction of window overlap [0, 1]
                    'hfo_name': "ripple"
                }
                
                rms_detector = RMSDetector(**kwargs_rmsdetector) 
                rms_detector = rms_detector.fit(raw)
                
                rms_hfo_df = rms_detector.df_
                hfo_dist_df = plot_events_hfo_2(rms_hfo_df['channels'], ch_info, raw._last_time)
                status = ['bad' if element in raw.info['bads'] else 'good' for element in hfo_dist_df['channels']]     
                hfo_dist_xarray = (hfo_dist_df.set_index(['channels']).to_xarray()).to_array()
                hfo_dist_xarray = hfo_dist_xarray.rename({'variable':'values'})    
                
                algorithms_params_array.append(hfo_dist_xarray)
                algorithms_params_names.append(f'bw-{combined_parameters[0]}_ww-{combined_parameters[1]}')
                

            subject_dataset = xr.Dataset(
                dict(
                    zip(algorithms_params_names, algorithms_params_array)
                    )
                )

            attributes = raw.info['subject_info']
            subject_dataset = subject_dataset.assign_attrs(**attributes)

            subject_datasets.append(subject_dataset)
        
    del raws #memory free
    
    return subject_datasets

if __name__ == "__main__":
    
    processed_data_directory = "data/processed/"
        
    kwargs_for_combine = {
                'lower_freq':   40,
                'upper_freq':   249,
                'band_width':   60,
                'window_min':   50,
                'window_max':   150,
                'window_step':  20
            }

    kwargs_bids = {
        'dataset':'data/bids/ds004100',
        'datatype':'ieeg',
        'task':'interictal',
        'acquisition':'seeg',
        'run':'01'
        }  
    
    # run process
    subject_datasets = data_processing(kwargs_bids, kwargs_for_combine) 
    
    # save results on individual subjects dataset -> 'subject_id'_procesed.pickle
    for dataset in subject_datasets:
        subject_id = dataset.attrs['his_id'] 
        filename = f"{subject_id}_processed.pickle"  
        
        with open(processed_data_directory + filename, 'wb') as file:
            pickle.dump(dataset, file)

    
    print(f'Hfo distribution analysis for {kwargs_bids["dataset"]} finished')

# %% grafica de todos las distribuciones de hfo encontradas para un subject

# for element in subject_dataset:
#     print(element)
    
#     df = subject_dataset[element].to_pandas().T.reset_index()
#     df = df.drop(df[df['colors'].isna()].index)
#     for bad in raw.info['bads']:
#         df = df.drop(df[df['channels'] == bad].index)
#     bar_chart(df)
