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

from util.general_utils import convert_to_bipolar

#set
# matplotlib.use('pdf')

#import data
# raw = mne.io.read_raw_edf('D:\Fernando_Icazatti\Trabajofinal\BBDD\Anexos_Pruebas\sub-HUP60\Reduced_sub-HUP060_ses-presurgery_task-interictal_acq-seeg_run-02_ieeg.edf') 

raw = mne.io.read_raw_edf('/home/proyectoepilepsia/Documents/Pruebas/CodeTesis/Reduced_sub-HUP060_ses-presurgery_task-interictal_acq-seeg_run-02_ieeg.edf', preload=True)

#read annotations from file
ann = raw.annotations

#create new annotations
ann.append(
    [2.5 , 25., 50., 107.5, 120.5],  #onset in seconds
    [2.5, 2.5, 2.5, 2.5, 2.5],     #duration 
    ['test1', 'test1', 'test1', 'test1', 'test1']) #description

#graphics of raw data and annotations
raw.plot(highpass=0.1, scalings=dict(eeg=40e-6)) 

raw = convert_to_bipolar(raw)
raw.plot(  )
raw.plot_psd(  )

event_id = {
    #'test1': 1, 
    'EventHfo': 2
    }

#converting annotations objects to event array
events,_ =mne.events_from_annotations(
    raw,
    event_id=event_id)
    # chunk_duration=30.)

# plot events
fig =mne.viz.plot_events(events, event_id=event_id,
                          sfreq=raw.info['sfreq'],
                          first_samp=events[0, 0])

#Events can also be plotted alongside the Raw object  
raw.plot(events=events, start=5, duration=100, color='gray',
         event_color={2: 'r'})

epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=0.9)
    
epochs.plot()


fmin = 8.
fmax = 13.
sfreq = raw.info['sfreq']  # the sampling frequency
con_methods = ['pli', 'coh', 'ciplv']
# con_methods = ['pli']
conn = []
for method in con_methods:
    conn.append(
        spectral_connectivity_epochs(
            data = epochs, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin,
            fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1
        )
    )

conn = np.array(conn)           

figure, axes = plt.subplots(1, 3, figsize=(8, 4), facecolor='black',
                         subplot_kw=dict(polar=True))

for ax, x in zip(axes, conn):  
    plot_connectivity_circle(x.get_data(output='dense')[:,:,0],
                            n_lines = 100,
                            node_names = raw.ch_names,
                            title='All-to-All Connectivity ' + x.method[0],
                            padding=0, fontsize_colorbar=6,
                            ax = ax)           

figure.show() # to see complete subplot

###### grafica de una sola matriz de conectividad a la vez
# for x in conn:
#     plot_connectivity_circle(x.get_data(output='dense')[:,:,0],
#                             node_names = raw.ch_names,
#                             title='All-to-All Connectivity ' + x.method[0])           

