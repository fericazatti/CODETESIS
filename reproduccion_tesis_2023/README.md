# Reproducción y robustez del resultado de la tesis (2023)

Reproduce la Tabla 4 de la tesis (Mann-Whitney del HRR entre pacientes con outcome S y F, 42
configuraciones del detector RMS) con el **pipeline original sin cambios de método**, y la evalúa en
cohortes y duraciones de registro ampliadas, como pidió el revisor de EJN-2026-01-33765.

## Qué es "el pipeline original"

`detect.py` usa el código de `main` (commit `35aa134`, copiado en `orig_src/`):

- Montaje bipolar original.
- Bandas (40,249) más seis sub-bandas de 60 Hz.
- `win_size` 50…150 paso 20 **en muestras**. El texto de la tesis/paper dice ms: 50 muestras ≈ 98 ms a
  512 Hz y ≈ 49 ms a 1024 Hz.
- Umbral 3 SD y overlap 0.25.

`analyze.py` calcula el HRR exactamente como `analysis.py`: excluye canales *bad*, toma el top
round(N_resect/2) y redondea a 2 decimales. El Mann-Whitney es bilateral, por aproximación normal con
corrección por empates y sin corrección de continuidad (equivale a la "Sig. asintótica" de SPSS).

**Única diferencia:** el código de 2023 obtenía el contacto ánodo con `split('-')[0]`, y eso falla
cuando los nombres llevan guion (HUP117: `AMFG-A1`). Acá se resuelve el ánodo real. Para nombres sin
guion el resultado es idéntico.

**Control:** con el primer registro, los 23 sujetos archivados en git dan **966/966 HRR idénticos a
2023** (23 × 42), y HUP133 da conteos idénticos en las 42 configuraciones.

## Cohortes

| Cohorte | Sujetos |
|---|---|
| `tesis24` | Los 24 de la Tabla 1 de la tesis (12 S / 12 F) |
| `tesis24+excl6` | Lo anterior más HUP130, 135, 139, 140, 141 y 142. Estaban procesados en 2023 y se borraron en el commit `6feaf92`; tienen outcome y etiquetas válidos |
| `todos` | Todos los SEEG con registro interictal procesables: 35. Se excluyen HUP132 y HUP173 (fs = 256 Hz, Nyquist 128 Hz, por debajo de las bandas originales). HUP150 queda fuera del HRR porque no tiene contactos etiquetados `resect` |

## Resultados (`results/`)

| Datos | Cohorte | n | C36 | C37 | C40 | Configs con p<0.05 | Mejor (q FDR) |
|---|---|---|---|---|---|---|---|
| Tesis (SPSS) | tesis24 | 12/12 | 0.05 | **0.04** | 0.06 | 1/42 | — |
| 2023 archivado* | tesis24 | 12/12 | 0.020 | **0.042** | 0.042 | 3/10 | — |
| Primer registro (5 min) | tesis24 | 12/12 | 0.054 | **0.042** (U=42.5) | 0.098 | 1/42 | C37 (0.69) |
| Primer registro | tesis24+excl6 | 17/13 | 0.098 | 0.139 | 0.247 | 0/42 | C36 (0.96) |
| Primer registro | todos | 18/16 | 0.151 | 0.099 | 0.341 | 0/42 | C37 (0.99) |
| Todos los registros (10 min) | tesis24 | 12/12 | **0.033** | 0.090 | 0.130 | 6/42 | C20 p=0.014 (0.27) |
| Todos los registros | tesis24+excl6 | 17/13 | 0.148 | 0.618 | 0.790 | 0/42 | C22 (0.55) |
| Todos los registros | todos | 18/16 | 0.103 | 0.693 | 0.824 | 0/42 | C22 (0.61) |

\* Usa los 23 pickles de git más los HRR de HUP117 que sobreviven en la salida de `analysis.ipynb`
(solo C33–C42). El pickle de HUP117 de 2023 no está en git y no se puede regenerar con el código
commiteado.

Ningún escenario sobrevive a la corrección por comparaciones múltiples: Bonferroni ×42 y FDR sobre 42.
Solo 4 de los 42 U de la Tabla 4 coinciden exactamente con los datos archivados, entre ellos C37. La
planilla que entró a SPSS no se archivó. `variants.py` prueba variantes del cálculo del HRR y ninguna
reproduce la tabla completa.

## Cómo correrlo

Se corrió en el servidor 151 (unas 1.3 h, pico de unos 34 GB):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./extract_2023.sh                 # pickles de 2023 desde el historial de git
./fetch_all.sh                    # ds004100 v1.1.3, solo SEEG interictal (~4 GB)
./launch.sh 20 12 34G             # completó la fase run01; en su fase allruns, 12 workers sobre
                                  # sujetos de 1024 Hz superaron el tope (OOM del propio servicio)
systemd-run --unit=hfo-ejn -p MemoryMax=38G -p OOMPolicy=continue -p CPUWeight=20 \
  -p WorkingDirectory=$PWD $PWD/run_allruns.sh 4 6   # allruns en 2 colas (pesados/livianos) + analyze.py
.venv/bin/python summary.py
```

`participants.tsv` del dataset trae `"sub-HUP151 "` con un espacio y finales de línea CRLF. En la copia
local se normalizó (original en `participants.tsv.orig`), porque si no `mne_bids` no encuentra a HUP151.
