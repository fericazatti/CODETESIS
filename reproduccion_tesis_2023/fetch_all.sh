#!/usr/bin/env bash
# Descarga en paralelo (12 sujetos a la vez) todos los sujetos del manifiesto.
cd "$(dirname "$0")"
mkdir -p logs
.venv/bin/python -c 'import json; print("\n".join(sorted(json.load(open("manifest_interictal.json")))))' \
  | xargs -P 12 -I{} nice -n 10 .venv/bin/python fetch.py {} > logs/fetch.log 2>&1
echo "FETCH_ALL_DONE $(date -Is)" >> logs/fetch.log
