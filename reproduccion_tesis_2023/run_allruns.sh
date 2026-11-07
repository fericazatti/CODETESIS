#!/usr/bin/env bash
# Fase 'allruns' con dos colas según la memoria que necesita cada sujeto:
# pesados (fs=1024 Hz) y livianos (fs<=512 Hz), en paralelo. Al final corre analyze.py.
# Uso: ./run_allruns.sh <workers_pesados> <workers_livianos>
set -u
cd "$(dirname "$0")"
WH=${1:-4}; WL=${2:-6}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python - <<'PY' > logs/allruns_queues.txt
import glob, pickle, os
for p in sorted(glob.glob("out/run01/*.pickle")):
    ds = pickle.load(open(p, "rb"))
    size = sum(os.path.getsize(f) for f in glob.glob(f"bids/ds004100/{ds.attrs['his_id']}/ses-presurgery/ieeg/*.edf"))
    print("H" if ds.attrs["sfreq"] > 600 else "L", size, ds.attrs["his_id"])
PY
job() { .venv/bin/python detect.py "$1" allruns > "logs/$1_allruns.log" 2>&1 \
          && echo "OK $1 allruns" >> logs/run_all.log || echo "FAIL $1 allruns" >> logs/run_all.log; }
export -f job
echo "START_ALLRUNS $(date -Is) pesados=$WH livianos=$WL" >> logs/run_all.log
awk '$1=="H"' logs/allruns_queues.txt | sort -k2 -rn | awk '{print $3}' | xargs -P "$WH" -I{} bash -c 'job {}' &
awk '$1=="L"' logs/allruns_queues.txt | sort -k2 -rn | awk '{print $3}' | xargs -P "$WL" -I{} bash -c 'job {}' &
wait
echo "MODE_DONE allruns $(date -Is)" >> logs/run_all.log
.venv/bin/python analyze.py > logs/analyze.log 2>&1
echo "ALL_DONE $(date -Is)" >> logs/run_all.log
