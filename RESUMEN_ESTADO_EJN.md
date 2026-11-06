# Estado de la Revisión EJN (EJN-2026-01-33765) — HFO & Pronóstico Quirúrgico

**Fecha de guardado:** 30 de Septiembre de 2026  
**Autor:** Fernando Esteban Icazatti  
**Ubicación del proyecto:** `/home/fernandoicazatti/Documents/personal/tesis-bioingenieria/codetesis`  
**Entorno Python:** `.venv` (activable con `source .venv/bin/activate`)

---

## 1. Estado Actual del Procesamiento

* **Cohorte:** **35 pacientes SEEG** de la base HUP (OpenNeuro `ds004100`).
* **Progreso de Cómputo:** **100% COMPLETADO (35/35 sujetos, 0 errores)**.
* **Archivos procesados:** Guardados en [`data/processed/sub-*_processed.pickle`](file:///home/fernandoicazatti/Documents/personal/tesis-bioingenieria/codetesis/data/processed/).
* **Configuraciones:**
  * **14 pacientes ($f_s = 1024$ Hz):** 54 configuraciones completas (High Gamma, Ripple y Fast Ripple).
  * **20 pacientes ($f_s = 512$ Hz):** 36 configuraciones (High Gamma y Ripple; Fast Ripple excluido por Nyquist $f_s/2 = 256$ Hz).
  * **1 paciente ($f_s = 256$ Hz):** 12 configuraciones (solo High Gamma; Ripple excluido por Nyquist $f_s/2 = 128$ Hz).
* **Multi-run:** Se concatenaron todos los runs interictales disponibles por paciente (reportados en [`data/results/interictal_runs_report.csv`](file:///home/fernandoicazatti/Documents/personal/tesis-bioingenieria/codetesis/data/results/interictal_runs_report.csv)).
* **Informe y Figuras Generadas:**
  * PDF consolidado: [`data/results/informe_hfo_ejn.pdf`](file:///home/fernandoicazatti/Documents/personal/tesis-bioingenieria/codetesis/data/results/informe_hfo_ejn.pdf)
  * Figuras en alta resolución: [`data/results/figures/`](file:///home/fernandoicazatti/Documents/personal/tesis-bioingenieria/codetesis/data/results/figures/) (F1 a F6).

---

## 2. Resumen Estadístico de los Hallazgos

| Banda | Rango [Hz] | Ventanas Evaluadas | Umbrales | N Sujetos | Mejor Configuración | AUC Outcome | Mann-Whitney $p$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fast Ripples** | 250–450 Hz | 3, 5, 10, 20, 30, 50 ms | 3, 5, 7 SD | 14 ($f_s=1024$) | `fr_w3_t5` (3 ms, 5 SD) | 0.622 | $p = 0.354$ |
| **High Gamma** | 65–100 Hz | 3, 5, 10, 20, 30, 50 ms | 3, 5, 7 SD | 35 | `hg_w50_t5` (50 ms, 5 SD) | 0.648 | $p = 0.114$ |
| **Ripples** | 100–240 Hz | 3, 5, 10, 20, 30, 50 ms | 3, 5, 7 SD | 34 ($f_s \ge 500$) | `rp_w50_t7` (50 ms, 7 SD) | 0.601 | $p = 0.298$ |

### Métricas Alternativas (Concordancia y Enriquecimiento):
* A nivel de **canal individual intra-paciente**, las Fast Ripples se encuentran significativamente enriquecidas dentro del área resecada:
  * **Concordancia Intra-Paciente (AUC canal resecado vs. no resecado):** AUC = **0.689** (`fr_w3_t5`).
  * **Ratio de Enriquecimiento:** AUC = **0.733** (`fr_w3_t5`).
* Sin embargo, la métrica agregada a nivel de paciente (**HRR**, HFO Resection Ratio) **no alcanza significancia estadística ($p > 0.11$)** para predecir si el paciente quedará libre de crisis (Engel I) o no.

---

## 3. Comparativa: Tesis Original (TF) vs. Revisión Actual (EJN)

| Dimensión | Trabajo Final de Grado (TF) | Revisión Actual EJN |
| :--- | :--- | :--- |
| **Documento base** | [`TF Fernando Icazatti-final.pdf`](file:///home/fernandoicazatti/Downloads/TF%20Fernando%20Icazatti-final.pdf) | Código refactorizado en `codetesis/` |
| **Pacientes** | $N=24$ (12 S vs. 12 F) | $N=35$ (todos los SEEG disponibles en HUP) |
| **Muestreo temporal** | Solo `run-01` (5 minutos) | Multi-run (todos los runs interictales concatenados) |
| **Bandas** | 7 sub-bandas de 60 Hz (40–100 a 190–249 Hz) | Bandas estándar: High Gamma (65–100), Ripple (100–240), Fast Ripple (250–450) |
| **Ventanas RMS** | 50, 70, 90, 110, 130, 150 ms | 3, 5, 10, 20, 30, 50 ms (incluye ultra-cortas) |
| **Conversión ventana** | `win_size` recibía ms como muestras enteras | `win_size = round(window_ms * fs / 1000)` formalmente |
| **Umbral** | Fijo en 3 SD | Barrido en 3, 5 y 7 SD |
| **Hallazgo clave** | C37 (190–249 Hz, $p=0.04$) y C36 ($p=0.05$) | Ninguna de las 54 configs alcanza $p<0.05$ (resultado negativo riguroso) |

---

## 4. Estrategia Editorial Acordada para la Resubmisión

Para conciliar el contenido previo de la tesis con las exigencias del revisor sin incurrir en contradicciones estadísticas:

1. **Estructura en dos fases en el paper:**
   * **Fase 1 (Exploración preliminar de la tesis):** Se reporta el análisis inicial en $N=24$ con ventanas de 50–150 ms y sub-bandas acotadas donde C36/C37 sugerían una correlación prometedora a nivel de un único run.
   * **Fase 2 (Validación metodológica exhaustiva motivada por la revisión):** Se somete la hipótesis al estándar más riguroso (cohorte completa $N=35$, concatenación multi-run, bandas estándar fisiológicas, ventanas ultra-cortas de 3–10 ms y umbrales estrictos de 5–7 SD).
2. **Conclusión científica de alto impacto:**
   * Demostrar de forma transparente que **el ajuste de parámetros de detección no rescata la capacidad predictiva del HRR interictal**.
   * Esto posiciona al manuscrito como un **estudio de benchmarking y reproducibilidad metodológica**, altamente valorado en revistas como EJN frente a estudios previos con sobreajuste o muestras reducidas.
3. **Respuesta formal a los revisores:**
   * Corrección de "FIR Butterworth" a FIR de fase cero con ventana Hamming y overlap-add en MNE.
   * Justificación biofísica de ventanas ultra-cortas (3–10 ms).
   * Incorporación formal del subanálisis de Fast Ripples en pacientes con $f_s = 1024$ Hz y advertencia metodológica sobre la restricción de Nyquist en SEEG clínico.
   * Inclusión del reporte de múltiples runs interictales.

---

## 5. Comandos para Retomar en la Próxima Sesión

* **Activar entorno:**
  ```bash
  cd /home/fernandoicazatti/Documents/personal/tesis-bioingenieria/codetesis
  source .venv/bin/activate
  ```
* **Regenerar reporte si fuera necesario:**
  ```bash
  python make_report.py
  ```
* **Ver resultados detallados de cualquier configuración:**
  ```bash
  python -c "import pandas as pd; print(pd.read_csv('data/results/interictal_runs_report.csv'))"
  ```
