"""
Métricas avanzadas de HFO a nivel canal y paciente (exploración del hallazgo).

Consolida la exploración que mostró que:
  - la hipótesis de SOLAPAMIENTO HFO<->resección no predice el outcome (nula), pero
  - features de la DISTRIBUCIÓN de HFO (residual fuera de la resección, focalidad)
    + cobertura sí predicen el outcome (LOO-CV AUC ~0.82), robustas a sueño/fs.

Usa las coordenadas fsaverage (espacio común entre pacientes) del dataset HUP para
construir una normativa espacial (leave-one-patient-out, kNN) y una "abnormalidad"
de HFO por canal = cuánto excede la tasa esperada en esa ubicación cerebral.
"""
import glob
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from src.utils.cloud_data import DEFAULT_BIDS_ROOT

RESECT = ["resect", "resect,soz"]
_RAW_BASE = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds004100/master"


# ----------------------------------------------------------------- electrodos (fsaverage)
def ensure_electrodes(subjects, bids_root=DEFAULT_BIDS_ROOT):
    """Descarga idempotente de los *_space-fsaverage_electrodes.tsv al caché BIDS."""
    for sub in subjects:
        sub = sub if str(sub).startswith("sub-") else f"sub-{sub}"
        dst_dir = os.path.join(bids_root, sub, "ses-presurgery", "ieeg")
        os.makedirs(dst_dir, exist_ok=True)
        fname = f"{sub}_ses-presurgery_acq-seeg_space-fsaverage_electrodes.tsv"
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(dst):
            continue
        url = f"{_RAW_BASE}/{sub}/ses-presurgery/ieeg/{fname}"
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception:
            pass
    return bids_root


def coord_map(subject, bids_root=DEFAULT_BIDS_ROOT):
    """dict contacto -> (x,y,z) en fsaverage (mm). {} si no hay archivo."""
    sub = subject if str(subject).startswith("sub-") else f"sub-{subject}"
    p = os.path.join(bids_root, sub, "ses-presurgery", "ieeg",
                     f"{sub}_ses-presurgery_acq-seeg_space-fsaverage_electrodes.tsv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p, sep="\t")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    m = {}
    for _, r in df.iterrows():
        try:
            m[str(r["name"])] = (float(r["x"]), float(r["y"]), float(r["z"]))
        except (ValueError, TypeError):
            pass
    return m


def _bip_coord(cm, chan):
    """Coord de un canal bipolar 'LA1-LA2' = punto medio de sus contactos."""
    pts = [cm[p] for p in str(chan).split("-") if p in cm]
    if not pts:
        return None
    c = tuple(np.mean(pts, axis=0))
    return c if all(np.isfinite(c)) else None


# ----------------------------------------------------------------- tabla canal-nivel
def channel_table(datasets, cfg, bids_root=DEFAULT_BIDS_ROOT):
    """
    Tabla larga (un canal por fila) para una config: sub, outcome, coords, rate,
    resect/soz/normal. Solo canales con coordenadas válidas.
    """
    rows = []
    for ds in datasets:
        if cfg not in set(map(str, ds.data_vars)):
            continue
        sub = str(ds.attrs["his_id"])
        cm = coord_map(sub, bids_root)
        outc = 1 if str(ds.attrs.get("outcome")) == "S" else 0
        df = ds[cfg].to_pandas().T.reset_index()
        for _, r in df.iterrows():
            c = _bip_coord(cm, r["channels"])
            if c is None:
                continue
            rate = pd.to_numeric(r["hfo_rate"], errors="coerce")
            rate = 0.0 if not np.isfinite(rate) else float(rate)
            st = str(r["status"])
            rows.append(dict(sub=sub, outcome=outc, x=c[0], y=c[1], z=c[2], rate=rate,
                             resect=int(st in RESECT), soz=int("soz" in st),
                             normal=int(st not in RESECT and "soz" not in st)))
    return pd.DataFrame(rows).dropna(subset=["x", "y", "z", "rate"])


def spatial_abnormality(T, K=20):
    """
    Anormalidad por canal = log2((rate+1)/(esperado+1)), donde 'esperado' es la
    mediana de la tasa de HFO de los K canales NORMALES más cercanos (en fsaverage)
    de OTROS pacientes (leave-one-patient-out, evita fuga de información).
    """
    T = T.copy()
    T["abn"] = np.nan
    for s in T["sub"].unique():
        ref = T[(T["sub"] != s) & (T["normal"] == 1)].dropna(subset=["x", "y", "z", "rate"])
        if len(ref) < K:
            continue
        nn = NearestNeighbors(n_neighbors=K).fit(ref[["x", "y", "z"]].values)
        cur = T[T["sub"] == s]
        _, idx = nn.kneighbors(cur[["x", "y", "z"]].values)
        exp = np.array([np.median(ref["rate"].values[ii]) for ii in idx])
        T.loc[cur.index, "abn"] = np.log2((cur["rate"].values + 1) / (exp + 1))
    return T


def channel_level_auc(T, score="rate"):
    """AUC (pooled) del score discriminando canales resecados vs no-resecados."""
    sub = T.dropna(subset=[score])
    return roc_auc_score(sub["resect"], sub[score])


# ----------------------------------------------------------------- features por paciente
def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n


def patient_features(datasets, cfg, bids_root=DEFAULT_BIDS_ROOT, state_csv=None):
    """
    Un paciente por fila, para una config. Features:
      resid    = media de abnormalidad en canales NO resecados (HFO residual)
      focality = Gini de la abnormalidad entre canales (concentración)
      global_rate = tasa media de HFO (todos los canales)
      coverage = nº de canales con coordenadas
      channel_auc = AUC intra-paciente rate -> resect
      sleep    = índice de estado (delta relativo) si se pasa state_csv
      outcome  = 1 (S) / 0 (F)
    """
    T = spatial_abnormality(channel_table(datasets, cfg, bids_root))
    state = None
    if state_csv and os.path.exists(state_csv):
        state = pd.read_csv(state_csv).set_index("subject")["sleep_delta"].to_dict()
    rows = []
    for s, g in T.groupby("sub"):
        g = g.dropna(subset=["abn"])
        if g["resect"].sum() == 0:
            continue
        non = g[g["resect"] == 0]
        try:
            ch_auc = roc_auc_score(g["resect"], g["rate"]) if g["resect"].nunique() == 2 else np.nan
        except Exception:
            ch_auc = np.nan
        rows.append(dict(
            sub=s, outcome=int(g["outcome"].iloc[0]),
            resid=non["abn"].mean() if len(non) else np.nan,
            focality=_gini(g["abn"].clip(lower=0)),
            global_rate=g["rate"].mean(),
            coverage=len(g),
            channel_auc=ch_auc,
            sleep=(state.get(s, np.nan) if state else np.nan),
        ))
    return pd.DataFrame(rows)
