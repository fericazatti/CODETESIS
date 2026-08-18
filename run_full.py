#!/usr/bin/env python3
"""
Orquestador de la corrida COMPLETA del detector de HFO sobre todos los sujetos
SEEG interictales del HUP (OpenNeuro ds004100), con control de memoria.

Diseñado para equipos con RAM ajustada:
  * Procesa DE A UN SUJETO y guarda su pickle inmediatamente (streaming).
  * RESUMIBLE: si el pickle ya existe, saltea (salvo --force).
  * Libera memoria entre sujetos (del + gc.collect()).
  * Guarda de RAM: lee MemAvailable de /proc/meminfo; si está bajo, espera y,
    si persiste, baja n_jobs a 1.
  * try/except por sujeto: uno malo no tumba el batch.
  * Log por sujeto a data/results/run_full.log y filas a interictal_runs_report.csv.

Uso:
    python run_full.py [--subjects HUP133 HUP157] [--njobs 3]
                       [--mem-floor-mb 1200] [--force] [--cleanup-edf]
                       [--config configs/detector_configs.tsv]
"""
import argparse
import csv
import gc
import os
import pickle
import shutil
import sys
import time
from datetime import datetime

import pandas as pd

from src.utils.cloud_data import ensure_dataset, DEFAULT_BIDS_ROOT, seeg_subjects
from src.utils.read_data_from_bids import detect_subjects
from data_processing import process_subject

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "data/results"
LOG_PATH = os.path.join(RESULTS_DIR, "run_full.log")
RUNS_CSV = os.path.join(RESULTS_DIR, "interictal_runs_report.csv")


def mem_available_mb():
    """MemAvailable (MB) desde /proc/meminfo, sin dependencias externas."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def log(msg):
    line = f"{datetime.now().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def wait_for_memory(floor_mb, max_wait_s=180):
    """
    Si MemAvailable < floor_mb, hace gc y espera hasta max_wait_s.
    Devuelve True si se recuperó memoria, False si sigue por debajo (=> bajar n_jobs).
    """
    avail = mem_available_mb()
    if avail is None or avail >= floor_mb:
        return True
    log(f"  [mem] MemAvailable={avail}MB < piso {floor_mb}MB -> gc + espera")
    waited = 0
    while waited < max_wait_s:
        gc.collect()
        time.sleep(15)
        waited += 15
        avail = mem_available_mb()
        if avail is not None and avail >= floor_mb:
            log(f"  [mem] recuperada: {avail}MB tras {waited}s")
            return True
    log(f"  [mem] sigue baja ({avail}MB) tras {waited}s -> se baja n_jobs")
    return False


def edf_dir(bids_root, subject):
    return os.path.join(bids_root, f"sub-{subject}", "ses-presurgery", "ieeg")


def append_run_row(row):
    """Append idempotente de una fila de run_report al CSV."""
    header = ["subject", "n_runs", "runs", "duration_min"]
    exists = os.path.exists(RUNS_CSV)
    with open(RUNS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in header})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="IDs sin prefijo (p.ej. HUP133). Default: todos los SEEG interictal.")
    ap.add_argument("--njobs", type=int, default=3, help="n_jobs del detector (default 3).")
    ap.add_argument("--mem-floor-mb", type=int, default=1200,
                    help="Piso de MemAvailable antes de esperar/bajar n_jobs.")
    ap.add_argument("--force", action="store_true", help="Reprocesa aunque exista el pickle.")
    ap.add_argument("--cleanup-edf", action="store_true",
                    help="Borra los EDF del sujeto tras procesarlo (baja disco, no RAM).")
    ap.add_argument("--config", default="configs/detector_configs.tsv")
    ap.add_argument("--bids-root", default=DEFAULT_BIDS_ROOT)
    ap.add_argument("--add-missing", action="store_true",
                    help="Modo incremental: calcula solo las configs que faltan en "
                         "el pickle existente y las fusiona (no reprocesa las ya hechas).")
    args = ap.parse_args()

    all_config_ids = set(pd.read_csv(args.config, sep="\t")["config_id"])

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t_start = time.time()
    log(f"=== run_full START | njobs={args.njobs} mem_floor={args.mem_floor_mb}MB "
        f"config={args.config} ===")

    # 1. Descarga selectiva (idempotente) de todos los SEEG interictal + metadatos.
    log("Conectando a OpenNeuro y asegurando dataset (descarga selectiva)...")
    bids_root = ensure_dataset(subjects=args.subjects, bids_root=args.bids_root)

    kwargs_bids = {'dataset': bids_root, 'datatype': 'ieeg',
                   'task': 'interictal', 'acquisition': 'seeg'}

    # 2. Sujetos a procesar = los detectados con run interictal SEEG.
    detected = detect_subjects(**kwargs_bids)
    if args.subjects:
        detected = [s for s in detected if s in set(args.subjects)]
    detected.sort()
    log(f"Sujetos SEEG interictal detectados: {len(detected)} -> {detected}")

    n_ok, n_skip, n_err = 0, 0, 0
    for i, subject in enumerate(detected, 1):
        pickle_path = os.path.join(PROCESSED_DIR, f"sub-{subject}_processed.pickle")
        existing = None
        only_configs = None

        if os.path.exists(pickle_path) and not args.force:
            if args.add_missing:
                with open(pickle_path, "rb") as f:
                    existing = pickle.load(f)
                only_configs = all_config_ids - set(map(str, existing.data_vars))
                if not only_configs:
                    log(f"[{i}/{len(detected)}] sub-{subject}: sin configs nuevas -> skip")
                    n_skip += 1
                    continue
            else:
                log(f"[{i}/{len(detected)}] sub-{subject}: pickle ya existe -> skip")
                n_skip += 1
                continue

        # Guarda de memoria antes de arrancar el sujeto.
        njobs = args.njobs
        if not wait_for_memory(args.mem_floor_mb):
            njobs = 1

        t0 = time.time()
        try:
            ds, row = process_subject(subject, kwargs_bids, args.config,
                                      n_jobs=njobs, only_configs=only_configs)
            if ds is None:
                log(f"[{i}/{len(detected)}] sub-{subject}: sin runs/configs nuevas -> skip")
                n_skip += 1
                continue

            if existing is not None:
                # Modo incremental: fusionar las configs nuevas al dataset existente.
                for var in ds.data_vars:
                    existing[var] = ds[var]
                ds = existing

            with open(pickle_path, "wb") as f:
                pickle.dump(ds, f)
            if existing is None:
                append_run_row(row)
            dt = time.time() - t0
            added = "+%d" % len(only_configs) if only_configs else str(len(ds.data_vars))
            log(f"[{i}/{len(detected)}] sub-{subject}: OK | njobs={njobs} "
                f"configs={added} total={len(ds.data_vars)} "
                f"MemAvail={mem_available_mb()}MB dt={dt:.0f}s")
            n_ok += 1

            if args.cleanup_edf:
                shutil.rmtree(edf_dir(bids_root, subject), ignore_errors=True)

            del ds
        except Exception as exc:  # noqa: BLE001 - un sujeto malo no tumba el batch
            n_err += 1
            log(f"[{i}/{len(detected)}] sub-{subject}: ERROR -> {type(exc).__name__}: {exc}")
        finally:
            gc.collect()

    log(f"=== run_full DONE | ok={n_ok} skip={n_skip} err={n_err} "
        f"total_min={(time.time()-t_start)/60:.1f} ===")


if __name__ == "__main__":
    sys.exit(main())
