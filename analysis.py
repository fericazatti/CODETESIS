# %% imports y definiciones de variables

import pickle
import os
import xarray as xr
import numpy as np

from src.utils.read_data_from_bids import read_four_subjects
from src.utils.plots.hfo_plots import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Configuración de rutas y processed_data_directorys
processed_data_directory = "data/processed"
results_directory = "data/results"

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
    
    

# %%

df_decode_detect = pd.DataFrame({'name':[], 'combination':[]})

for i, element in enumerate(ds.coords["algorithm_params"].values):
    df = ds.sel(algorithm_params = element).to_pandas().T
    df_drop = df.drop("algorithm_params")
    
    df_process = df_process.join(df_drop['hrr']).rename(columns={'hrr': f'hfo_detect_C{i+1}'})
    df_decode_detect = df_decode_detect.append({'name': f'hfo_detect_C{i+1}', 'combination': element}, ignore_index=True)

    # df_drop[['hrr','outcome']].plot.scatter(x='hrr',y='outcome').set_title(element)



# %% obtener dataframe con el resultado del proceso
id = []
age = []
sex = []
hand = []
outcome = []
engel = []
therapy = []
implant = []
target = []
lesion_status = []
age_onset = [] 

for dataset in datasets:
    id.append(dataset.attrs['his_id'])
    age.append(dataset.attrs['age'])
    sex.append(dataset.attrs['sex'])
    hand.append(dataset.attrs['hand'])
    outcome.append(dataset.attrs['outcome'])
    engel.append(dataset.attrs['engel'])
    therapy.append(dataset.attrs['therapy'])
    implant.append(dataset.attrs['implant'])
    target.append(dataset.attrs['target'])
    lesion_status.append(dataset.attrs['lesion_status'])
    age_onset.append(dataset.attrs['age_onset'])

dict(
    zip(id,
    zip(age,
    zip(sex,
    zip(hand,
    zip(outcome,
    zip(engel,
    zip(therapy,
    zip(implant,
    zip(target,
    zip(lesion_status,
    age_onset)))))))))))
dictonary = {
    nombre_lista: lista for nombre_lista, lista in zip([
        'id',
        'age',
        'sex',
        'hand',
        'outcome',
        'engel',
        'therapy',
        'implant',
        'target',
        'lesion_status',
        'age_onset'],
        [id,
        age,
        sex,
        hand,
        outcome,
        engel,
        therapy,
        implant,
        target,
        lesion_status,
        age_onset])}

df_process = pd.DataFrame(dictonary)
df_process = df_process.set_index('id')
# %% get hrr results vs outcome 

df = ds.sel(algorithm_params = 'bw-(160, 220)_ww-110').to_pandas().T
df_drop = df.drop("algorithm_params")
df_drop[['hrr','outcome']].plot.scatter(x='hrr',y='outcome').set_title('bw-(130, 190)_ww-90')

# %% optimizacion de parametros de regresión logistica

train = df_drop
# definiendo input y output
X_train = np.array(train['hrr']).reshape((-1, 1))
Y_train = np.array(train['outcome'])
Y_train = Y_train.astype(int)

# creando modelo
model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_train)
print(classification_report(Y_train, Y_pred))

# imprimiendo parametros
print(f"intercepto (b): {model.intercept_}")
print(f"pendiente (w): {model.coef_}")

b = model.intercept_
w = model.coef_

# puntos de la recta
x = np.linspace(0,train['hrr'].max(),1000)
y = 1/(1+np.exp(-(w*x+b)))

# grafica de la recta
train.plot.scatter(x='hrr',y='outcome')
plt.plot(x, y.reshape(1000), '-r')
plt.ylim(0,train['outcome'].max()*1.1)
# plt.grid()
plt.show()

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

        # Fit the classifier
        clf = LogisticRegression(C=1e5)
        clf.fit(X_train, Y_train)

        # and plot the result
        plt.figure(1, figsize=(6, 3))
        plt.clf()
        plt.scatter(X_train.ravel(), Y_train, label="example data", color="blue", zorder=20)
        X_test = np.linspace(-0.5, 1, 300)

        loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
        plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)

        # ols = LinearRegression()
        # ols.fit(X_train, Y_train)
        # plt.plot(
        #     X_test,
        #     ols.coef_ * X_test + ols.intercept_,
        #     label="Linear Regression Model",
        #     linewidth=1,
        # )
        plt.axhline(0.5, color=".5")

        plt.ylabel("outcome")
        plt.xlabel("hrr")
        plt.xticks([0, 0.25, 0.5, 0,75])
        plt.yticks([0, 0.5, 1])
        plt.ylim(-0.25, 1.25)
        plt.xlim(-0.2, 0.75)
        plt.legend(
            
            fontsize="small",
        )
        plt.tight_layout()
        plt.title( combination.iloc[0])
        plt.show()

# %%

