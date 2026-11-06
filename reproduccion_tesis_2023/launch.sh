#!/usr/bin/env bash
# Lanza run_all.sh como servicio transitorio 'hfo-ejn' con límites para proteger los
# servicios del servidor (PetBrain, Orthanc, MONAI). Detener: systemctl stop hfo-ejn
# Uso: ./launch.sh <workers_run01> <workers_allruns> <memory_max, ej. 34G>
cd "$(dirname "$0")"
systemd-run --unit=hfo-ejn --collect \
  -p WorkingDirectory="$PWD" -p MemoryMax="${3:-34G}" -p MemorySwapMax=0 \
  -p CPUWeight=20 -p Nice=19 -p IOSchedulingClass=idle \
  "$PWD/run_all.sh" "${1:-20}" "${2:-12}"
