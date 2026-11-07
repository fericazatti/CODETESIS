#!/usr/bin/env bash
# Monitor de memoria sin dependencias.
# Cada INTERVAL s escribe una fila a data/results/mem_monitor.csv:
#   timestamp, mem_available_mb, mem_used_mb, swap_used_mb, run_rss_mb
# run_rss_mb = RSS del proceso `python run_full.py` (0 si no está corriendo).
#
# Uso:  bash tools/mem_monitor.sh [INTERVAL_SEG]     (default 15)
# Frená con: kill <pid>  (o borrando data/results/mem_monitor.stop)

set -u
INTERVAL="${1:-15}"
OUT="data/results/mem_monitor.csv"
STOP="data/results/mem_monitor.stop"
mkdir -p data/results

if [ ! -f "$OUT" ]; then
    echo "timestamp,mem_available_mb,mem_used_mb,swap_used_mb,run_rss_mb" > "$OUT"
fi

while true; do
    [ -f "$STOP" ] && { rm -f "$STOP"; break; }

    ts=$(date -Is)

    # /proc/meminfo en kB -> MB
    mem_avail=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
    mem_total=$(awk '/^MemTotal:/{print int($2/1024)}' /proc/meminfo)
    mem_used=$(( mem_total - mem_avail ))
    swap_total=$(awk '/^SwapTotal:/{print int($2/1024)}' /proc/meminfo)
    swap_free=$(awk '/^SwapFree:/{print int($2/1024)}' /proc/meminfo)
    swap_used=$(( swap_total - swap_free ))

    # RSS (kB->MB) del/los proceso(s) run_full.py
    run_rss=$(ps -eo rss,args 2>/dev/null | awk '/[r]un_full\.py/{s+=$1} END{print int(s/1024)}')
    run_rss=${run_rss:-0}

    echo "${ts},${mem_avail},${mem_used},${swap_used},${run_rss}" >> "$OUT"
    sleep "$INTERVAL"
done
