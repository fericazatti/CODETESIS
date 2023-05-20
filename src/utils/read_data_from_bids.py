# %%
# Imports needs for read data
from openneuro import download
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

# %% define functions
def detect_subjects(dataset,
                    datatype,
                    task,
                    acquisition,
                    run):
    """
    Purpose: 
    """
    bids_root = f'{dataset}' 
    session = get_entity_vals(bids_root, 'session')
    bids_path = BIDSPath(subject= None,
                session= session[0],
                task= task,                      
                root=bids_root,                     
                datatype= datatype,
                acquisition= acquisition, 
                run= run) 
    
    subjects_id = []
    for path in bids_path.match(): subjects_id.append(path.subject)
    subjects_id = list(set(subjects_id)) #delete repeats values
    subjects_id.sort()

    return subjects_id

def read_four_subjects( subjects,  
                    dataset,    
                    datatype,
                    task,   
                    acquisition,
                    run):
    """
    Purpose: 
    """
    bids_root = f'{dataset}' 
    session = get_entity_vals(bids_root, 'session')
    bids_path = []
    subjects = [element for element in subjects if element is not None] # delete None elements for matrix subjects       
    for subject in subjects:    
        bids_path.append( BIDSPath(subject= subject,
                        session= session[0],
                        task= task,                      
                        root=bids_root,                     
                        datatype= datatype,
                        acquisition= acquisition,
                        run = run))


    # lectura de los archivos raw e info de los canales (tsv) para cada uno de los registros
    raws = []
    channels_info_paths = []
    extra_params = {'preload' : True}
    for path in bids_path:    
        for value in path.match():
            if value.extension == '.edf':
                raws.append(read_raw_bids(bids_path=value, verbose=False, extra_params = extra_params))
            
            if value.extension == '.tsv':
                path_file = f'{dataset}/sub-{value.subject}/ses-{value.session}/{value.datatype}/{value.basename}'
                channels_info_paths.append(path_file)
    return raws, channels_info_paths

 # end def



    # %%
if __name__ == '__main__':
    dataset = 'ds004100'

    # download(dataset = dataset)
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
    # extra_params = {'preload' : True}

    for path in bids_path.match():    
        if path.extension == '.edf':
            raws.append(read_raw_bids(bids_path=path, verbose=False))
        
        if path.extension == '.tsv':
            path_file = f'{dataset}/sub-{path.subject}/ses-{path.session}/{path.datatype}/{path.basename}'
            cahnnels_info_paths.append(path_file)

    subjects_id = []
    for path in bids_path.match(): subjects_id.append(path.subject)
    subjects_id = list(set(subjects_id)) #delete repeats values
    subjects_id.sort()

    # %% leer los subject y crear la matriz de subjects

    kwargs = {
        'dataset':'data/bids',
        'datatype':'ieeg',
        'task':'interictal',
        'acquisition':'seeg',
        'run':'01'
    }

    subjects_id = detect_subjects(**kwargs)

    # Calculamos el número de filas necesarias
    rows = len(subjects_id) // 4 + (1 if len(subjects_id) % 4 != 0 else 0)
    # Creamos la matriz bidimensional con valores None
    subjects_matrix = [[None] * 4 for _ in range(rows)]

    # Llenamos la matriz con los elementos del arreglo
    for i, element in enumerate(subjects_id):
        row = i // 4
        column = i % 4
        subjects_matrix[row][column] = element

    # %% llamada a la funcion de lectura raws



    for subjects in subjects_matrix:
        subjects = [element for element in subjects if element is not None] # delete None elements for matrix subjects       
        read_four_subjects(subjects=subjects, **kwargs)
        
        


    # %%
