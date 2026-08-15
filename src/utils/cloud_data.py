"""
Conexión automática al dataset en la nube (OpenNeuro ds004100 - HUP iEEG).

En lugar de asumir que el dataset ya está en disco (como hacía el pipeline
original con la ruta local ``data/bids/ds004100``), este módulo descarga
**solo lo necesario** desde OpenNeuro la primera vez y deja un caché BIDS local
que el resto del pipeline consume sin cambios.

Se descargan únicamente:
    - participants.tsv / participants.json   (para saber qué sujetos son SEEG y su outcome)
    - sub-*/ses-presurgery/ieeg/*task-interictal*acq-seeg*   (EDF + JSON + channels.tsv + electrodes.tsv)

La descarga es idempotente: openneuro-py saltea los archivos ya presentes, así
que correr esto de nuevo no vuelve a bajar nada.

Dataset: https://openneuro.org/datasets/ds004100  (v1.1.3, 58 sujetos, fs=500 Hz)
"""

import os
import pandas as pd

DATASET = "ds004100"
DEFAULT_BIDS_ROOT = os.path.join("data", "bids", DATASET)


def _download(include, bids_root):
    """Descarga selectiva vía openneuro-py. ``include`` = lista de globs BIDS."""
    try:
        from openneuro import download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Se requiere 'openneuro-py' para la descarga automática "
            "(`pip install openneuro-py`). Alternativa manual: "
            "`aws s3 cp --no-sign-request --recursive "
            "s3://openneuro.org/ds004100/ <destino>`."
        ) from exc

    os.makedirs(bids_root, exist_ok=True)
    download(dataset=DATASET, target_dir=bids_root, include=include, verify_hash=False)


def _ensure_participants(bids_root):
    """Garantiza participants.tsv/json en el caché y devuelve el DataFrame."""
    participants_tsv = os.path.join(bids_root, "participants.tsv")
    if not os.path.exists(participants_tsv):
        _download(["participants.tsv", "participants.json"], bids_root)
    return pd.read_csv(participants_tsv, sep="\t")


def seeg_subjects(bids_root=DEFAULT_BIDS_ROOT):
    """
    Lista de sujetos con implante SEEG (columna ``implant`` de participants.tsv),
    sin el prefijo 'sub-' (formato que espera mne-bids: 'HUP157').
    """
    participants = _ensure_participants(bids_root)
    seeg = participants.loc[
        participants["implant"].astype(str).str.upper() == "SEEG", "participant_id"
    ]
    return sorted(s.replace("sub-", "") for s in seeg)


def ensure_dataset(subjects=None, bids_root=DEFAULT_BIDS_ROOT):
    """
    Asegura que estén en el caché local los archivos interictales SEEG de los
    sujetos pedidos, descargando desde OpenNeuro solo lo que falte.

    Parameters
    ----------
    subjects : list[str] | None
        IDs de sujeto sin prefijo (p.ej. ['HUP157', 'HUP160']). Si es None, se
        usan todos los sujetos SEEG del dataset (según participants.tsv).
    bids_root : str
        Carpeta destino del caché BIDS local. Se devuelve para pasarla al lector.

    Returns
    -------
    str
        La raíz BIDS local (``bids_root``).
    """
    # 1. participants + descubrimiento de sujetos SEEG
    if subjects is None:
        subjects = seeg_subjects(bids_root)
    else:
        _ensure_participants(bids_root)  # asegura metadatos aunque se pasen sujetos
        subjects = [s.replace("sub-", "") for s in subjects]

    # 2. patrones include: solo interictal SEEG (EDF + sidecars) por sujeto
    include = ["participants.tsv", "participants.json"]
    for sub in subjects:
        include.append(f"sub-{sub}/*/ieeg/*task-interictal*acq-seeg*")

    _download(include, bids_root)
    return bids_root
