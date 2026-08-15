"""
Carga y validación de la tabla de configuraciones del detector de HFO.

Reemplaza al generador procedural ``algorithm_params`` (src/utils/hfo_2.py).
Cada fila del TSV es UNA configuración del detector. El usuario edita ese
archivo (configs/detector_configs.tsv) para definir todas las combinaciones a
barrer, sin tocar el código.

Columnas esperadas:
    config_id     identificador único (nombra la variable de salida)
    band_label    'highgamma' | 'ripple'   (fast ripples no son válidos a fs=500)
    f_low         límite inferior de banda [Hz]
    f_high        límite superior de banda [Hz]   (debe ser < fs/2)
    window_ms     ancho de ventana RMS [ms]       (se convierte a muestras con fs)
    overlap_pct   solape de ventana [%]           (0-100)
    threshold_sd  multiplicador de desvíos estándar del umbral (p.ej. 3/5/7)

La conversión ms -> muestras se hace acá (arreglo del bug histórico en el que
``win_size`` se pasaba en ms directamente como muestras sin usar ``fs``).
"""

import pandas as pd

VALID_BANDS = {"highgamma", "ripple"}
MIN_WINDOW_SAMPLES = 4  # ventana RMS mínima representable de forma significativa

REQUIRED_COLUMNS = [
    "config_id",
    "band_label",
    "f_low",
    "f_high",
    "window_ms",
    "overlap_pct",
    "threshold_sd",
]


def ms_to_samples(window_ms, fs):
    """Convierte un ancho de ventana en ms a número entero de muestras a ``fs``."""
    return int(round(window_ms * fs / 1000.0))


def load_configs(path, fs):
    """
    Lee la tabla TSV de configuraciones y la valida contra la frecuencia de
    muestreo ``fs`` del registro que se va a procesar.

    Parameters
    ----------
    path : str
        Ruta al TSV (p.ej. 'configs/detector_configs.tsv').
    fs : float
        Frecuencia de muestreo [Hz] del registro. Se usa para (a) validar
        Nyquist y (b) convertir ``window_ms`` a ``window_samples``.

    Returns
    -------
    pandas.DataFrame
        La tabla validada, con una columna extra ``window_samples`` (int).

    Raises
    ------
    ValueError
        Si faltan columnas, hay ``config_id`` duplicados, alguna banda supera
        Nyquist (f_high >= fs/2), f_low >= f_high, un band_label inválido, o la
        ventana resulta en menos de ``MIN_WINDOW_SAMPLES`` muestras a ``fs``.
    """
    df = pd.read_csv(path, sep="\t")

    # 0. columnas presentes
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")

    if df.empty:
        raise ValueError(f"La tabla de configuraciones {path} está vacía.")

    # tipos
    for col in ["f_low", "f_high", "window_ms", "overlap_pct", "threshold_sd"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # 1. config_id únicos
    dups = df["config_id"][df["config_id"].duplicated()].tolist()
    if dups:
        raise ValueError(f"config_id duplicados en {path}: {dups}")

    # 2. band_label válido
    bad_band = df.loc[~df["band_label"].isin(VALID_BANDS), "config_id"].tolist()
    if bad_band:
        raise ValueError(
            f"band_label inválido (debe ser uno de {sorted(VALID_BANDS)}) "
            f"en configs: {bad_band}"
        )

    # 3. banda coherente
    bad_order = df.loc[df["f_low"] >= df["f_high"], "config_id"].tolist()
    if bad_order:
        raise ValueError(f"Se requiere f_low < f_high en configs: {bad_order}")

    # 4. Nyquist: toda banda debe quedar estrictamente por debajo de fs/2
    nyquist = fs / 2.0
    over_nyq = df.loc[df["f_high"] >= nyquist, "config_id"].tolist()
    if over_nyq:
        raise ValueError(
            f"f_high >= Nyquist ({nyquist:.1f} Hz) para configs: {over_nyq}. "
            f"A fs={fs} Hz no se pueden analizar esas bandas (p.ej. fast ripples). "
            f"Subí fs o quitá esas filas."
        )

    # 5. solape en rango
    bad_ov = df.loc[(df["overlap_pct"] < 0) | (df["overlap_pct"] >= 100), "config_id"].tolist()
    if bad_ov:
        raise ValueError(f"overlap_pct debe estar en [0, 100) en configs: {bad_ov}")

    # 6. ventana -> muestras suficiente
    df["window_samples"] = (df["window_ms"] * fs / 1000.0).round().astype(int)
    thin = df.loc[df["window_samples"] < MIN_WINDOW_SAMPLES, "config_id"].tolist()
    if thin:
        raise ValueError(
            f"Ventana demasiado corta para fs={fs} Hz "
            f"(< {MIN_WINDOW_SAMPLES} muestras) en configs: {thin}. "
            f"Aumentá window_ms."
        )

    return df.reset_index(drop=True)
