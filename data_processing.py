# %% Imports
import os
import pickle

import pandas as pd
import xarray as xr

from mne_hfo import RMSDetector

from src.utils.preprocessing import convert_to_bipolar
from src.utils.hfo_2 import plot_events_hfo_2
from src.utils.config_table import load_configs
from src.utils.cloud_data import ensure_dataset, DEFAULT_BIDS_ROOT
from src.utils.read_data_from_bids import detect_subjects, read_subjects, participant_attrs


# %%
def process_subject(subject, kwargs_bids, config_path, n_jobs=-1, only_configs=None):
    """
    Procesa UN sujeto con las configuraciones de la tabla ``config_path``.

    Concatena todos sus runs interictales, arma el montaje bipolar, valida la
    tabla contra la fs real y corre el RMSDetector por cada config.

    Parameters
    ----------
    n_jobs : int
        Paralelismo del RMSDetector. -1 = todos los cores. Se acota (p.ej. 1-3)
        cuando hay presión de memoria (ver run_full.py).
    only_configs : set[str] | None
        Si se da, se calculan SOLO esas config_id (modo incremental para agregar
        configs nuevas a un pickle existente). None = todas las de la tabla.

    Returns
    -------
    subject_dataset : xarray.Dataset | None   (None si el sujeto no tiene runs)
    run_report_row : dict | None
    """
    raws, channels, run_report = read_subjects(subjects=[subject], **kwargs_bids)
    if not raws:
        return None, None

    raw = convert_to_bipolar(raws[0])
    fs = raw.info['sfreq']

    # La tabla se valida contra la fs real. drop_invalid=True descarta las configs
    # no representables a esta fs (p.ej. ripple a fs=256) y corre las válidas.
    configs = load_configs(config_path, fs=fs, drop_invalid=True)
    if only_configs is not None:
        configs = configs[configs['config_id'].isin(only_configs)]
    if len(configs) == 0:
        return None, None
    ch_info = pd.read_csv(channels[0], sep='\t')

    config_names = []
    config_arrays = []
    for cfg in configs.itertuples(index=False):
        kwargs_rmsdetector = {
            'filter_band': (float(cfg.f_low), float(cfg.f_high)),
            'threshold': float(cfg.threshold_sd),   # nº de desvíos estándar
            'win_size': int(cfg.window_samples),    # ms -> muestras (ya convertido)
            'overlap': float(cfg.overlap_pct) / 100.0,
            'hfo_name': cfg.band_label,
            'n_jobs': n_jobs,
        }
        rms_detector = RMSDetector(**kwargs_rmsdetector).fit(raw)
        rms_hfo_df = rms_detector.df_

        hfo_dist_df = plot_events_hfo_2(rms_hfo_df['channels'], ch_info, raw._last_time)
        hfo_dist_xarray = hfo_dist_df.set_index(['channels']).to_xarray().to_array()
        hfo_dist_xarray = hfo_dist_xarray.rename({'variable': 'values'})

        config_arrays.append(hfo_dist_xarray)
        config_names.append(cfg.config_id)

    subject_dataset = xr.Dataset(dict(zip(config_names, config_arrays)))
    # Atributos clínicos desde participants.tsv (robusto a la versión de mne_bids).
    subject_dataset = subject_dataset.assign_attrs(
        **participant_attrs(kwargs_bids['dataset'], subject)
    )
    return subject_dataset, run_report[0]


def data_processing(kwargs_bids, config_path, n_jobs=-1):
    """
    Procesa TODOS los sujetos detectados. Envoltura sobre ``process_subject``.
    (Para corridas grandes con control de memoria, ver run_full.py.)
    """
    subjects_id = detect_subjects(**kwargs_bids)
    subject_datasets = []
    full_run_report = []
    for subject in subjects_id:
        ds, row = process_subject(subject, kwargs_bids, config_path, n_jobs=n_jobs)
        if ds is None:
            continue
        subject_datasets.append(ds)
        full_run_report.append(row)
    return subject_datasets, full_run_report


if __name__ == "__main__":

    processed_data_directory = "data/processed/"
    results_directory = "data/results/"
    os.makedirs(processed_data_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)

    config_path = "configs/detector_configs.tsv"

    # 1. Conexión automática al dataset en la nube: descarga selectiva (solo
    #    interictal SEEG + metadatos) al caché local si no está ya presente.
    bids_root = ensure_dataset(subjects=None, bids_root=DEFAULT_BIDS_ROOT)

    kwargs_bids = {
        'dataset': bids_root,
        'datatype': 'ieeg',
        'task': 'interictal',
        'acquisition': 'seeg',
    }

    # 2. Procesamiento
    subject_datasets, run_report = data_processing(kwargs_bids, config_path)

    # 3. Guardado por sujeto -> 'subject_id'_processed.pickle
    for dataset in subject_datasets:
        subject_id = dataset.attrs['his_id']
        filename = f"{subject_id}_processed.pickle"
        with open(processed_data_directory + filename, 'wb') as file:
            pickle.dump(dataset, file)

    # 4. Reporte de runs interictales por sujeto (para responder al revisor:
    #    "how many interictal runs were considered")
    run_df = pd.DataFrame(run_report)
    run_df.to_csv(results_directory + "interictal_runs_report.csv", index=False)

    print(f'HFO distribution analysis for {bids_root} finished')
    if not run_df.empty:
        print(f'Runs interictales por sujeto -> {results_directory}interictal_runs_report.csv')
        print(run_df.to_string(index=False))
