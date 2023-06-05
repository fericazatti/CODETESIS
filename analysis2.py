# %%
import pickle
import os
import xarray as xr
import numpy as np

from src.utils.read_data_from_bids import read_four_subjects
from src.utils.plots.hfo_plots import *
from src.utils.analysis_utils import organize_results

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score 
from scipy.special import expit, logit

from src.utils.plots.hfo_plots import regression_plot

# Configuración de rutas y processed_data_directorys
processed_data_directory = "data/processed"
results_directory = "data/results"

kwargs_bids = {
    'dataset':'data/bids/ds004100',
    'datatype':'ieeg',
    'task':'interictal',
    'acquisition':'seeg',
    'run':'01'
    } 

# %% leer datos del procesamiento de un sujeto especifico

subject = "HUP117"
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

raws, channels = read_four_subjects([subject] ,**kwargs_bids)

for element in subject_dataset:
    print(element)
    
    df = subject_dataset[element].to_pandas().T.reset_index()
    df = df.drop(df[df['colors'].isna()].index)
    for bad in raws[0].info['bads']:
        df = df.drop(df[df['channels'] == bad].index)
    bar_chart(df)

del raws

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

        condicion = df['status'].isin(['resect', 'resect,soz'])
        filtrado = df[condicion]
        contadores = filtrado['status'].value_counts()

        resection_size = contadores.sum()
        resection_size = round(resection_size/2) 
        
        hfo_region = df.sort_values('counts', ascending=False).head(resection_size)
        condicion = hfo_region['status'].isin(['resect', 'resect,soz'])
        filtrado = hfo_region[condicion]
        contadores = filtrado['status'].value_counts()

        hfo_in_resection = contadores.sum()

        #hrr = round((hfo_in_resection/resection_size),2) + np.random.random() * 0.1 - 0.05#hfo resection ratio
        hrr = round((hfo_in_resection/resection_size),2)
        hrr_list.append(hrr)
                
        if subject_dataset.attrs['outcome'] == 'S':
            outcomes.append(1)
        else:
            outcomes.append(0)

    fusion = list(zip(hrr_list, outcomes))
    fusion_lista = [list(fila) for fila in fusion]

    ds[subject_dataset.attrs['his_id']] = xr.DataArray(fusion_lista, dims=('algorithm_params', 'statisicians'))
    
    df_process = organize_results(datasets)

# %%

df_decode_detect = pd.DataFrame({'name':[], 'combination':[]})

for i, element in enumerate(ds.coords["algorithm_params"].values):
    df = ds.sel(algorithm_params = element).to_pandas().T
    df_drop = df.drop("algorithm_params")
    
    df_process = df_process.join(df_drop['hrr']).rename(columns={'hrr': f'hfo_detect_C{i+1}'})
    df_decode_detect = df_decode_detect.append({'name': f'hfo_detect_C{i+1}', 'combination': element}, ignore_index=True)

    # df_drop[['hrr','outcome']].plot.scatter(x='hrr',y='outcome').set_title(element)

# %% get hrr results vs outcome 

# df = ds.sel(algorithm_params = 'bw-(160, 220)_ww-110').to_pandas().T
# df_drop = df.drop("algorithm_params")
# df_drop[['hrr','outcome']].plot.scatter(x='hrr',y='outcome').set_title('bw-(130, 190)_ww-90')

# %% global analysis of logistic regression
df_results = {
    'combination_name': [],
    'band_width':[],
    'window_width':[],
    'accuracy': [],    
    'f1-score': [],
    'AUC': []
}
for value in df_decode_detect.iterrows():
    value = value[1]
    
    df = ds.sel(algorithm_params = value['combination']).to_pandas().T
    df_drop = df.drop("algorithm_params")
    
    train = df_drop
    # definiendo input y output
    X_train = np.array(train['hrr']).reshape((-1, 1))
    Y_train = np.array(train['outcome'])
    Y_train = Y_train.astype(int)

    # creando modelo
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_train)
    report_dict = classification_report(Y_train, Y_pred, output_dict=True)
    
    #AUC calculo
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_train, y_pred_proba)
    auc = roc_auc_score(Y_train, y_pred_proba)   
        
    #agregar valores
    df_results['combination_name'].append(value['name'])
    df_results['band_width'].append(None)
    df_results['window_width'].append(None)
    df_results['accuracy'].append(report_dict['accuracy'])
    df_results['f1-score'].append(report_dict['1']['f1-score'])
    df_results['AUC'].append(auc)
    
df_results = pd.DataFrame(df_results)
# %% Prueba plot de sickit learn 

producto = df_results['accuracy'] * df_results['f1-score']
delete_repeat = set(producto)
ordered = sorted(list(delete_repeat))
maximos = ordered[-3:]
indices_maximo = []
for maximo in maximos:
    indices_maximo.append(np.where(producto == maximo)[0].tolist())

for row in indices_maximo:
    for maximo in row:
        
        name = df_results.iloc[maximo]['combination_name']
        combination = df_decode_detect.loc[df_decode_detect['name'] == name]['combination']
        
        df = ds.sel(algorithm_params = combination.iloc[0]).to_pandas().T
        df_drop = df.drop("algorithm_params")
        train = df_drop
        
        regression_plot(train, combination)
        
        # definiendo input y output
        X_train = np.array(train['hrr']).reshape((-1, 1))                
        Y_pred = model.predict(X_train)
        print(classification_report(Y_train, Y_pred))
        

# %%
