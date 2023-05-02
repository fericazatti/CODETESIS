# %%
# Imports needs for read data
from openneuro import download
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

# %%

dataset = 'ds004100'

download(dataset = dataset)
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


bids_path = BIDSPath(subject= None,
                session= session[0],
                task= task,                      
                root=bids_root,                     
                datatype= datatype,
                acquisition= acquisition) 


# %% lectura de los archivos raw e info de los canales (tsv) para cada uno de los registros
raws = []
cahnnels_info_paths = []
extra_params = {'preload' : True}

for path in bids_path.match():    
    if path.extension == '.edf':
        raws.append(read_raw_bids(bids_path=path, verbose=False, extra_params = extra_params))
    
    if path.extension == '.tsv':
        path_file = f'{dataset}/sub-{path.subject}/ses-{path.session}/{path.datatype}/{path.basename}'
        cahnnels_info_paths.append(path_file)

# %%
