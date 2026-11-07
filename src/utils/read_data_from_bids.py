# %%
# Imports needs for read data
import os
import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals


# Campos clínicos/demográficos de participants.tsv que el análisis downstream
# espera en los attrs del dataset (his_id, outcome, engel, ...).
def participant_attrs(dataset, subject):
    """
    Lee participants.tsv y devuelve la fila del sujeto como dict de atributos,
    mapeando 'participant_id' -> 'his_id'.

    Necesario porque mne_bids >=0.14 ya NO vuelca las columnas custom de
    participants.tsv (outcome, engel, therapy, target, lesion_status, age_onset)
    dentro de raw.info['subject_info']: las descarta con un warning. Se leen acá
    de forma robusta a la versión de mne_bids.
    """
    pid = subject if subject.startswith('sub-') else f'sub-{subject}'
    participants = pd.read_csv(os.path.join(dataset, 'participants.tsv'), sep='\t')
    row = participants.loc[participants['participant_id'] == pid]
    if row.empty:
        return {'his_id': pid}
    attrs = row.iloc[0].to_dict()
    attrs['his_id'] = attrs.pop('participant_id')
    return attrs


# %% define functions
def detect_subjects(dataset,
                    datatype,
                    task,
                    acquisition,
                    run=None):
    """
    Devuelve la lista de sujetos del BIDS que cumplen los filtros dados.

    ``run`` queda como parámetro opcional por compatibilidad, pero por defecto
    (None) se enumeran los sujetos sin fijar el run, de modo que aparezcan
    también los que no tienen 'run-01'.
    """
    bids_root = f'{dataset}'
    session = get_entity_vals(bids_root, 'session')
    bids_path = BIDSPath(subject=None,
                         session=session[0],
                         task=task,
                         root=bids_root,
                         datatype=datatype,
                         acquisition=acquisition,
                         run=run)

    subjects_id = []
    for path in bids_path.match():
        subjects_id.append(path.subject)
    subjects_id = list(set(subjects_id))  # delete repeats values
    subjects_id.sort()

    return subjects_id


def list_interictal_runs(subject, dataset, datatype, task, acquisition, session):
    """
    Enumera todos los runs (EDF) de un sujeto para el task/acquisition dados,
    ordenados por etiqueta de run. Devuelve una lista de BIDSPath (uno por EDF).
    """
    bids_path = BIDSPath(subject=subject,
                         session=session,
                         task=task,
                         root=dataset,
                         datatype=datatype,
                         acquisition=acquisition)
    edf_paths = [p for p in bids_path.match() if p.extension == '.edf']
    edf_paths.sort(key=lambda p: (p.run is None, p.run))
    return edf_paths


def _concatenate_runs(edf_paths):
    """
    Lee y concatena todos los runs de un sujeto en un único Raw.

    Los runs se alinean al conjunto de canales COMÚN a todos (por si algún run
    tiene canales distintos) antes de concatenar. Devuelve (raw, run_labels).
    """
    extra_params = {'preload': True}
    run_raws = []
    run_labels = []
    for path in edf_paths:
        raw = read_raw_bids(bids_path=path, verbose=False, extra_params=extra_params)
        run_raws.append(raw)
        run_labels.append(path.run)

    if len(run_raws) == 1:
        return run_raws[0], run_labels

    # canales comunes a todos los runs (preserva el orden del primero)
    common = set(run_raws[0].ch_names)
    for raw in run_raws[1:]:
        common &= set(raw.ch_names)
    common_ordered = [ch for ch in run_raws[0].ch_names if ch in common]
    for raw in run_raws:
        raw.pick(common_ordered)

    concatenated = mne.concatenate_raws(run_raws, verbose=False)
    return concatenated, run_labels


def read_subjects(subjects,
                  dataset,
                  datatype,
                  task,
                  acquisition,
                  run=None):
    """
    Lee un grupo de sujetos concatenando TODOS sus runs interictales.

    A diferencia de la versión previa (que fijaba run='01' y usaba un solo run),
    acá se enumeran todos los runs del task/acquisition y se concatenan por
    sujeto. ``run`` se ignora (queda por compatibilidad de firma).

    Returns
    -------
    raws : list[mne.io.Raw]
        Un Raw concatenado por sujeto.
    channels_info_paths : list[str]
        Ruta al channels.tsv del primer run de cada sujeto (SOZ/resect son
        consistentes entre runs del mismo sujeto).
    run_report : list[dict]
        Un dict por sujeto: {subject, n_runs, runs, duration_min}.
    """
    session = get_entity_vals(dataset, 'session')[0]
    subjects = [s for s in subjects if s is not None]  # limpia matriz con None

    raws = []
    channels_info_paths = []
    run_report = []

    for subject in subjects:
        edf_paths = list_interictal_runs(subject, dataset, datatype, task, acquisition, session)
        if not edf_paths:
            continue

        raw, run_labels = _concatenate_runs(edf_paths)
        raws.append(raw)

        # channels.tsv del primer run
        first = edf_paths[0]
        ch_tsv = (f'{dataset}/sub-{first.subject}/ses-{first.session}/'
                  f'{first.datatype}/'
                  f'{first.copy().update(suffix="channels", extension=".tsv").basename}')
        channels_info_paths.append(ch_tsv)

        run_report.append({
            'subject': subject,
            'n_runs': len(edf_paths),
            'runs': run_labels,
            'duration_min': round(raw.n_times / raw.info['sfreq'] / 60.0, 2),
        })

    return raws, channels_info_paths, run_report


# Alias de compatibilidad con el nombre histórico usado por otros scripts.
# Nota: ahora devuelve una TERCERA salida (run_report). Los llamadores deben
# desempacar tres valores: raws, channels, run_report = read_four_subjects(...)
def read_four_subjects(subjects, dataset, datatype, task, acquisition, run=None):
    return read_subjects(subjects, dataset, datatype, task, acquisition, run)
