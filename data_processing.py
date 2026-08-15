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
def data_processing(kwargs_bids, config_path):
    """
    Procesa cada sujeto SEEG con TODAS las configuraciones definidas en la tabla
    externa ``config_path`` (TSV). Reemplaza al generador procedural de 42
    combinaciones: ahora el barrido lo define el usuario editando el TSV.

    Correcciones respecto de la versión previa:
      * ``win_size`` se calcula como round(window_ms * fs / 1000) usando la fs
        real del registro (antes se pasaban los ms directamente como muestras).
      * ``threshold`` (nº de STD) es una columna barrible del TSV (antes fijo=3).
      * Se concatenan TODOS los runs interictales por sujeto (antes solo run-01).

    Returns
    -------
    subject_datasets : list[xarray.Dataset]
    run_report : list[dict]   (nº de runs y minutos analizados por sujeto)
    """
    subjects_id = detect_subjects(**kwargs_bids)

    subject_datasets = []
    full_run_report = []

    # Se procesa de a un sujeto (lectura + concatenación de sus runs interictales)
    for subject in subjects_id:
        raws, channels, run_report = read_subjects(subjects=[subject], **kwargs_bids)
        if not raws:
            continue
        full_run_report.extend(run_report)

        raw = convert_to_bipolar(raws[0])
        fs = raw.info['sfreq']

        # La tabla de configs se valida contra la fs real del registro:
        # rechaza bandas que superan Nyquist (p.ej. fast ripples a fs=500) y
        # ventanas demasiado cortas para esa fs.
        configs = load_configs(config_path, fs=fs)

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
            }

            rms_detector = RMSDetector(**kwargs_rmsdetector).fit(raw)
            rms_hfo_df = rms_detector.df_

            hfo_dist_df = plot_events_hfo_2(rms_hfo_df['channels'], ch_info, raw._last_time)
            hfo_dist_xarray = hfo_dist_df.set_index(['channels']).to_xarray().to_array()
            hfo_dist_xarray = hfo_dist_xarray.rename({'variable': 'values'})

            config_arrays.append(hfo_dist_xarray)
            config_names.append(cfg.config_id)

        subject_dataset = xr.Dataset(dict(zip(config_names, config_arrays)))
        # Atributos clínicos desde participants.tsv (his_id, outcome, engel, ...)
        # -> robusto a la versión de mne_bids, que ya no los pone en subject_info.
        subject_dataset = subject_dataset.assign_attrs(
            **participant_attrs(kwargs_bids['dataset'], subject)
        )
        subject_datasets.append(subject_dataset)

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
