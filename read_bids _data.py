# %%
# Imports
# -------
# We are importing everything we need for this example:
import os
import os.path as op
import openneuro

from mne.datasets import sample
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report,
                      find_matching_paths, get_entity_vals, inspect_dataset)

# %%
# Download a subject's data from an OpenNeuro BIDS dataset
# --------------------------------------------------------
#
# Download the data, storing each in a ``target_dir`` target directory, which,
# in ``mne-bids`` terminology, is the `root` of each BIDS dataset. This example
# uses this `EEG dataset <https://openneuro.org/datasets/ds002778>`_ of
# resting-state recordings of patients with Parkinson's disease.
#

# .. note: If the keyword argument include is left out of
#          ``openneuro.download``, the whole dataset will be downloaded.
#          We're just using data from one subject to reduce the time
#          it takes to run the example.

dataset = 'ds004100'
subject = 'HUP139'

# subjects = ['sub-HUP060', 'sub-HUP064', 'sub-HUP112', 'sub-HUP116']

# Download one subject's data from each dataset
# bids_root = op.join(op.dirname(sample.data_path()), dataset)
# if not op.isdir(bids_root):
#     os.makedirs(bids_root)

openneuro.download(dataset=dataset, include=[f'sub-{subject}'])
# openneuro.download(dataset=dataset, include= subjects)
bids_root = f'./{dataset}' 

# bids_root = '/home/proyectoepilepsia/mne_data/ds004100'

# %%
#Acceder a un sujeto/os dentro de la bbdd
#Luego crea los objetos path que apuntan a cada uno de los archivos de interes
#Finalmente se crea un arreglo con todos los objetos RAW listos para ser procesados

session = get_entity_vals(bids_root, 'session')
datatype = 'ieeg'
extensions = ['.edf']
task = 'interictal'
acquisition = 'seeg'

bids_path = BIDSPath(subject= None,
                     session= session[0],
                     task= task,                      
                     root=bids_root,                     
                     datatype= datatype,
                     acquisition= acquisition)


raws = []
for value in bids_path.match():
    if value.extension == '.edf':
        raws.append(read_raw_bids(bids_path=value, verbose=False, preload = True))

# %% procesar objetos raw

        


