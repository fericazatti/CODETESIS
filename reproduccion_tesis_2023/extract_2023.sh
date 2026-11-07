#!/usr/bin/env bash
# Extrae del historial de git los resultados de detección de 2023 que usa analyze.py:
#   pickles_2023/tesis/          -> los 24 pickles de main (commit 35aa134)
#   pickles_2023/excluidos_2023/ -> los 6 sujetos borrados en el commit 6feaf92 (desde 531955f)
set -e
cd "$(dirname "$0")"
mkdir -p pickles_2023/tesis pickles_2023/excluidos_2023
for f in $(git ls-tree --name-only 35aa134 ../data/processed/); do
  git show "35aa134:data/processed/$(basename "$f")" > "pickles_2023/tesis/$(basename "$f")"
done
for s in 130 135 139 140 141 142; do
  git show "531955f:data/processed/sub-HUP${s}_processed.pickle" > "pickles_2023/excluidos_2023/sub-HUP${s}_processed.pickle"
done
ls pickles_2023/*/ | wc -l
