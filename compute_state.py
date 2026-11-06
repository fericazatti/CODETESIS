#!/usr/bin/env python3
"""
Estima un índice de ESTADO (sueño NREM vs vigilia) por paciente desde la señal
cruda interictal, para chequear si la tasa global de HFO (asociada al outcome)
está confundida por el estado de registro.

Índice = potencia relativa en delta (1-4 Hz) sobre 1-45 Hz, mediana entre canales
SEEG. NREM -> más delta. Se calcula sobre la señal REFERENCIAL (no bipolar: el
montaje bipolar atenúa las ondas lentas de campo amplio).

Salida: data/results/state_index.csv  (subject, sleep_delta, n_ch, sfreq)
"""
import glob, os, pickle, time
import numpy as np, pandas as pd

from src.utils.cloud_data import DEFAULT_BIDS_ROOT
from src.utils.read_data_from_bids import read_subjects

OUT = "data/results/state_index.csv"
LOG = "data/results/state.log"


def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def rel_delta(raw):
    # excluir canales tipo EKG/ECG por nombre (vienen mal etiquetados como SEEG)
    picks = [c for c in raw.ch_names if not c.upper().startswith(("EKG", "ECG"))]
    raw = raw.copy().pick(picks)
    sf = raw.info["sfreq"]
    psd = raw.compute_psd(method="welch", fmin=1, fmax=45,
                          n_fft=int(2 * sf), verbose=False)
    p = psd.get_data()                     # (n_ch, n_freqs)
    f = psd.freqs
    delta = p[:, (f >= 1) & (f < 4)].sum(axis=1)
    total = p[:, (f >= 1) & (f < 45)].sum(axis=1)
    rd = delta / np.where(total > 0, total, np.nan)
    return float(np.nanmedian(rd)), len(picks), sf


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", default=DEFAULT_BIDS_ROOT)
    args = ap.parse_args()
    kb = {"dataset": args.bids_root, "datatype": "ieeg",
          "task": "interictal", "acquisition": "seeg"}

    subs = sorted(os.path.basename(f).replace("_processed.pickle", "").replace("sub-", "")
                  for f in glob.glob("data/processed/sub-*_processed.pickle"))
    log(f"=== state START | {len(subs)} sujetos ===")
    rows = []
    for i, s in enumerate(subs, 1):
        try:
            raws, _, _ = read_subjects(subjects=[s], **kb)
            if not raws:
                continue
            rd, nch, sf = rel_delta(raws[0])
            rows.append(dict(subject=f"sub-{s}", sleep_delta=round(rd, 4), n_ch=nch, sfreq=sf))
            log(f"[{i}/{len(subs)}] sub-{s}: rel_delta={rd:.3f} nch={nch} sf={sf}")
            del raws
            import gc; gc.collect()
        except Exception as e:
            log(f"[{i}/{len(subs)}] sub-{s}: ERROR {type(e).__name__}: {e}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    log(f"=== state DONE | {len(rows)} filas -> {OUT} ===")


if __name__ == "__main__":
    main()
