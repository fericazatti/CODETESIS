"""Detector HFO con el pipeline ORIGINAL de la tesis (2023), sin cambios de método.

Igual que data_processing.py @ main (commit 35aa134):
  - montaje bipolar original (src/utils/preprocessing.convert_to_bipolar),
  - 42 configuraciones: bandas (40,249) + sub-bandas de 60 Hz, win_size 50..150 paso 20
    EN MUESTRAS (así lo pasaba el código original a RMSDetector),
  - threshold = 3 SD, overlap = 0.25,
  - tabla de eventos por canal = plot_events_hfo_2 original.

Única diferencia (documentada): el código de 2023 obtenía el contacto ánodo con
channel.split('-')[0], que falla si los nombres de contacto llevan guion (p.ej. HUP117:
'AMFG-A1'). Acá se resuelve el ánodo real; para nombres sin guion el resultado es idéntico.

Uso: python detect.py HUP133 run01|allruns
Salida: out/<modo>/sub-<ID>.pickle  (resumible: si existe, no recalcula)
"""
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "orig_src"))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from mne_bids import BIDSPath, read_raw_bids  # noqa: E402
from mne_hfo import RMSDetector  # noqa: E402

from src.utils.hfo_2 import algorithm_params  # noqa: E402  (grilla original)
from src.utils.preprocessing import convert_to_bipolar  # noqa: E402  (montaje original)

BIDS = os.path.join(HERE, "bids", "ds004100")
GRID = dict(lower_freq=40, upper_freq=249, band_width=60,
            window_min=50, window_max=150, window_step=20)


def anode_of(channel, names):
    first = channel.split("-")[0]
    if first in names:  # caso original (nombres sin guion): idéntico a 2023
        return first
    for n in sorted(names, key=len, reverse=True):
        if channel.startswith(n + "-") and channel[len(n) + 1:] in names:
            return n
    return first


def events_table(det_channels, ch_info, last_time, unresolved):
    """plot_events_hfo_2 original, con resolución de ánodo para nombres con guion."""
    names = set(ch_info["name"])
    channels = det_channels.unique()
    counts = np.array([])
    ch_names_last = []
    for channel in channels:
        counts = np.append(counts, len(det_channels[det_channels == channel]))
        ch_names_last.append(anode_of(channel, names))
    status_desc = []
    for channel in ch_names_last:
        match = ch_info.loc[ch_info["name"] == channel, "status_description"]
        if len(match):
            status_desc.append(match.iloc[0])
        else:  # en 2023 esto era un IndexError
            unresolved.add(channel)
            status_desc.append(np.nan)
    hfo_rate = counts / last_time * 60
    df = pd.DataFrame({"channels": channels, "ch_split": ch_names_last, "counts": counts,
                       "status": status_desc, "hfo_rate": hfo_rate.round()})
    status = df["status"].unique()
    df = df.assign(colors=[""] * df.shape[0])
    color = 0x4E4E4E
    for state in status:
        for i in range(df.shape[0]):
            if str(df.at[i, "status"]).casefold() == str(state).casefold():
                df.at[i, "colors"] = "#" + hex(color)[2:]
        color = color + 0x006456
    return df


def run_paths(sub):
    base = BIDSPath(subject=sub, session="presurgery", task="interictal", acquisition="seeg",
                    datatype="ieeg", root=BIDS, suffix="ieeg", extension=".edf")
    runs = sorted(p.run for p in base.copy().update(run=None).match() if p.extension == ".edf")
    return [base.copy().update(run=r) for r in runs]


def main(sub, mode):
    out_dir = os.path.join(HERE, "out", mode)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"sub-{sub}.pickle")
    if os.path.exists(out):
        print(f"{sub} {mode}: ya existe, skip", flush=True)
        return
    t0 = time.time()
    paths = run_paths(sub)
    if mode == "run01":
        paths = [p for p in paths if p.run == "01"]
    if not paths:
        raise RuntimeError(f"{sub}: sin runs interictales para modo {mode}")

    raws, bads = [], set()
    for p in paths:
        r = read_raw_bids(p, verbose=False, extra_params={"preload": True})
        bads |= set(r.info["bads"])
        raws.append(r)
    if len(raws) > 1:  # canales comunes a todos los runs, en el orden del primero
        common = set(raws[0].ch_names).intersection(*[set(r.ch_names) for r in raws[1:]])
        keep = [c for c in raws[0].ch_names if c in common]
        for r in raws:
            r.pick(keep)
            r.info["bads"] = [b for b in bads if b in keep]
        raw = mne.concatenate_raws(raws, verbose=False)
    else:
        raw = raws[0]
    del raws  # libera los runs individuales (solo memoria; no cambia el cálculo)
    sfreq, last_time = raw.info["sfreq"], raw._last_time
    print(f"{sub} {mode}: runs={[p.run for p in paths]} sfreq={sfreq} dur={last_time:.0f}s "
          f"nch={len(raw.ch_names)}", flush=True)

    raw = convert_to_bipolar(raw)
    ch_info = pd.read_csv(str(paths[0].copy().update(suffix="channels", extension=".tsv")),
                          sep="\t")
    names, arrs, unresolved = [], [], set()
    for band, ww in algorithm_params(**GRID):
        det = RMSDetector(filter_band=band, threshold=3, win_size=ww, overlap=0.25,
                          hfo_name="ripple", n_jobs=1, verbose=False).fit(raw)
        d = events_table(det.df_["channels"], ch_info, last_time, unresolved)
        arrs.append(d.set_index(["channels"]).to_xarray().to_array().rename({"variable": "values"}))
        names.append(f"bw-{band}_ww-{ww}")

    part = pd.read_csv(os.path.join(BIDS, "participants.tsv"), sep="\t", dtype=str)
    part["participant_id"] = part["participant_id"].str.strip()
    row = part.set_index("participant_id").loc[f"sub-{sub}"].str.strip()
    ds = xr.Dataset(dict(zip(names, arrs))).assign_attrs(
        his_id=f"sub-{sub}", outcome=row["outcome"], engel=row["engel"], therapy=row["therapy"],
        mode=mode, runs=json.dumps([p.run for p in paths]), sfreq=float(sfreq),
        duration_s=float(last_time), bads=json.dumps(sorted(bads)),
        unresolved=json.dumps(sorted(unresolved)))
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(ds, f)
    os.replace(tmp, out)
    print(f"{sub} {mode}: DONE {time.time() - t0:.0f}s unresolved={sorted(unresolved)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].replace("sub-", "").strip(), sys.argv[2])
