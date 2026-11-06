#!/usr/bin/env bash
# Corre detect.py para todos los sujetos (más pesados primero) y ambos modos.
# Pensado para lanzarse como servicio transitorio con límites (ver launch.sh).
# Uso: ./run_all.sh <workers_run01> <workers_allruns>
set -u
cd "$(dirname "$0")"
W1=${1:-20}
W2=${2:-12}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
# sujetos ordenados por tamaño total de EDF (desc) para balancear la carga
SUBS=$(for d in bids/ds004100/sub-*; do echo "$(du -sb "$d" | cut -f1) $(basename "$d")"; done | sort -rn | awk '{print $2}')
echo "START $(date -Is) workers=$W1/$W2 n_subs=$(echo "$SUBS" | wc -l)" >> logs/run_all.log
for MODE in run01 allruns; do
  [ "$MODE" = run01 ] && W=$W1 || W=$W2
  printf '%s\n' $SUBS | xargs -P "$W" -I{} sh -c \
    ".venv/bin/python detect.py {} $MODE > logs/{}_$MODE.log 2>&1 && echo 'OK {} $MODE' >> logs/run_all.log || echo 'FAIL {} $MODE' >> logs/run_all.log"
  echo "MODE_DONE $MODE $(date -Is)" >> logs/run_all.log
done
.venv/bin/python analyze.py > logs/analyze.log 2>&1
echo "ALL_DONE $(date -Is)" >> logs/run_all.log
