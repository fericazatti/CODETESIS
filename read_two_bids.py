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

dataset = 'ds004100'
bids_root = f'./{dataset}' 

# %%
#Acceder a un sujeto/os dentro de la bbdd
#Luego crea los objetos path que apuntan a cada uno de los archivos de interes
#Finalmente se crea un arreglo con todos los objetos RAW listos para ser procesados

session = get_entity_vals(bids_root, 'session')
datatype = 'ieeg'
extensions = ['.edf']
task = 'interictal'
acquisition = 'seeg'

subjects = ['HUP060', 'HUP135']

bids_path = []
for subject in subjects:    
    bids_path.append( BIDSPath(subject= subject,
                        session= session[0],
                        task= task,                      
                        root=bids_root,                     
                        datatype= datatype,
                        acquisition= acquisition) )


raws = []
extra_params = {'preload' : True}
for path in bids_path:    
    for value in path.match():
        if value.extension == '.edf':
            raws.append(read_raw_bids(bids_path=value, verbose=False, extra_params = extra_params))

# %% procesar objetos raw

        


