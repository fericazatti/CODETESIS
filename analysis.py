# %% imports y definiciones de variables

import pickle
import os
import xarray as xr
import numpy as np

from src.utils.read_data_from_bids import read_four_subjects
from src.utils.plots.hfo_plots import *

# Configuración de rutas y processed_data_directorys
processed_data_directory = "data/processed"
results_directory = "data/results"

# %% leer datos del procesamiento de un sujeto especifico

subject = "HUP148"
file_path = f'data/processed/sub-{subject}_processed.pickle'


with open(file_path, "rb") as file:
    # Load the object from the pickle file
    subject_dataset = pickle.load(file)

# %% leer todos los datos procesados obtenidos
# Obtiene la lista de archivos en el processed_data_directory
files = os.listdir(processed_data_directory)

datasets = []
for archivo in files:
    if archivo.endswith(".pickle"):
        ruta_archivo = os.path.join(processed_data_directory, archivo)

        with open(ruta_archivo, "rb") as file:
            data = pickle.load(file)
    datasets.append(data)
    
# %% gráficas de la distribucion de hfo
kwargs_bids = {
    'dataset':'data/bids/ds004100',
    'datatype':'ieeg',
    'task':'interictal',
    'acquisition':'seeg',
    'run':'01'
    } 

raw, channels = read_four_subjects([subject] ,**kwargs_bids)

for element in subject_dataset:
    print(element)
    
    df = subject_dataset[element].to_pandas().T.reset_index()
    df = df.drop(df[df['colors'].isna()].index)
    for bad in raw[0].info['bads']:
        df = df.drop(df[df['ch_split'] == bad].index)
    bar_chart(df)

del raw

# %% leer todos los datos procesados obtenidos
# Obtiene la lista de archivos en el processed_data_directory
files = os.listdir(processed_data_directory)

datasets = []
for archivo in files:
    if archivo.endswith(".pickle"):
        ruta_archivo = os.path.join(processed_data_directory, archivo)

        with open(ruta_archivo, "rb") as file:
            data = pickle.load(file)
    datasets.append(data)
# cálculo de estadisticos

# Crea un diccionario vacío para almacenar los datos
data = {}

# Define las coordenadas vacías
data['statisicians'] = []
data['algorithm_params'] = []

# Crea un objeto Dataset vacío con las coordenadas especificadas
ds = xr.Dataset(data)

#asignacion de coordenadas
ds = ds.assign_coords(statisicians = ['hrr', 'outcome'])

subject_dataset = datasets[0]
for element in subject_dataset:
    new_param = element
    ds = ds.assign_coords(algorithm_params=ds['algorithm_params'].values.tolist() + [new_param])



for subject_dataset in datasets:
    
    subject_id = subject_dataset.attrs['his_id']
    raw, channels = read_four_subjects([subject_id.replace('sub-', '')] ,**kwargs_bids)
    
    hrr_list = []
    outcomes = []

    for element in subject_dataset:
        
        df = subject_dataset[element].to_pandas().T.reset_index()
        df = df.drop(df[df['colors'].isna()].index)
        for bad in raw[0].info['bads']:
            df = df.drop(df[df['ch_split'] == bad].index)    

        condicion = df['status'].isin(['resect', 'resect,soz', 'soz'])
        filtrado = df[condicion]
        contadores = filtrado['status'].value_counts()

        resection_size = contadores.sum()
        resection_size = round(resection_size)
        
        hfo_region = df.sort_values('counts', ascending=False).head(resection_size)
        condicion = hfo_region['status'].isin(['resect', 'resect,soz'])
        filtrado = hfo_region[condicion]
        contadores = filtrado['status'].value_counts()

        hfo_in_resection = contadores.sum()

        hrr = round((hfo_in_resection/resection_size),2) + np.random.random() * 0.1 - 0.05#hfo resection ratio

        hrr_list.append(hrr)
        
        if subject_dataset.attrs['outcome'] == 'S':
            outcomes.append(1)
        else:
            outcomes.append(0)

    fusion = list(zip(hrr_list, outcomes))
    fusion_lista = [list(fila) for fila in fusion]

    ds[subject_dataset.attrs['his_id']] = xr.DataArray(fusion_lista, dims=('algorithm_params', 'statisicians'))

# %%

for element in ds.coords["algorithm_params"].values:
    df = ds.sel(algorithm_params = element).to_pandas().T
    df_drop = df.drop("algorithm_params")
    df_drop[['hrr','outcome']].plot.scatter(x='hrr',y='outcome').set_title(element)



# %%
