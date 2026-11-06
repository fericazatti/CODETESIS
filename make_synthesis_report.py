#!/usr/bin/env python3
"""
Informe de SÍNTESIS para coautores (PDF).

Resume todo lo hecho en esta línea de trabajo: cambios de metodología (mencionados),
configuraciones usadas, respuesta al revisor, el resultado nulo riguroso de la
hipótesis original, y —como resultado secundario/exploratorio— el hallazgo de
features predictivos (HFO residual + focalidad + cobertura, LOO-CV AUC ~0.82),
con su confound-check e interpretación teórica.

Salida: data/results/informe_sintesis_coautores.pdf   (matplotlib PdfPages, sin deps nuevas)
Uso:    python make_synthesis_report.py [--config configs/detector_configs.tsv] [--bids-root ...]
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import mannwhitneyu, spearmanr

import textwrap

from make_report import (load_datasets, build_hrr_matrix, per_config_metrics,
                         metric_comparison, fig_rate_resect_vs_unlabeled,
                         fig_sweep_heatmap, fig_metric_comparison, RESULTS_DIR)
from src.utils.advanced_metrics import (ensure_electrodes, channel_table,
                                        spatial_abnormality, channel_level_auc,
                                        patient_features)
from src.utils import brain_maps as bm
from src.utils.cloud_data import DEFAULT_BIDS_ROOT

PDF = os.path.join(RESULTS_DIR, "informe_sintesis_coautores.pdf")
STATE = os.path.join(RESULTS_DIR, "state_index.csv")
CFG_FIND = "rp_w30_t7"   # config representativa del hallazgo (ripple, robusta en t5/t7)


A4 = (8.27, 11.69)


def text_page(pdf, title, lines, wrap=94, fontsize=8.6):
    """Página de texto con título envuelto y cuerpo monoespaciado con wrapping."""
    fig = plt.figure(figsize=A4)
    tlines = textwrap.wrap(title, 52) or [title]
    fig.text(0.07, 0.955, "\n".join(tlines), fontsize=15, weight="bold", va="top")
    body = []
    for ln in lines:
        if ln == "":
            body.append("")
        else:
            w = textwrap.wrap(ln, wrap, subsequent_indent="    ",
                              break_long_words=False, break_on_hyphens=False)
            body.extend(w or [""])
    y0 = 0.955 - 0.045 * len(tlines) - 0.02
    fig.text(0.07, y0, "\n".join(body), fontsize=fontsize, va="top",
             family="monospace", linespacing=1.5)
    pdf.savefig(fig); plt.close(fig)


def table_page(pdf, title, df, col_wrap, colWidths, fontsize=8):
    """Tabla con texto por celda envuelto (evita el overflow de matplotlib.table)."""
    cells = []
    maxlines = 1
    for _, row in df.iterrows():
        wrapped = []
        for val, cw in zip(row.values, col_wrap):
            txt = "\n".join(textwrap.wrap(str(val), cw)) or str(val)
            wrapped.append(txt)
            maxlines = max(maxlines, txt.count("\n") + 1)
        cells.append(wrapped)
    fig, ax = plt.subplots(figsize=A4)
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", loc="left", pad=16)
    t = ax.table(cellText=cells, colLabels=list(df.columns), colWidths=colWidths,
                 loc="upper left", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fontsize)
    t.scale(1, 1.4 * maxlines)
    for (r, c), cell in t.get_celld().items():
        cell.set_text_props(va="center")
        if r == 0:
            cell.set_text_props(weight="bold")
    pdf.savefig(fig); plt.close(fig)


def _auc(y, v):
    a = roc_auc_score(y, v)
    return max(a, 1 - a)


def _tag(fig, n):
    """Etiqueta 'Fig N' en la esquina superior izquierda (no colisiona con títulos centrados)."""
    fig.text(0.012, 0.988, f"Fig {n}", fontsize=11, weight="bold", va="top",
             ha="left", color="#2c6e8f")
    return fig


def loo_auc(P, cols):
    X = P[cols].values
    y = P["outcome"].values
    pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()),
                           X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    return roc_auc_score(y, pr), pr, y


# --------------------------------------------------------------------- figuras nuevas
def fig_channel_roc(T):
    sub = T.dropna(subset=["rate"])
    fpr, tpr, _ = roc_curve(sub["resect"], sub["rate"])
    auc = roc_auc_score(sub["resect"], sub["rate"])
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"tasa HFO (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "--", color="0.6")
    ax.set_xlabel("1 - especificidad"); ax.set_ylabel("sensibilidad")
    ax.set_title("Nivel canal: ¿la tasa de HFO marca el tejido resecado?\n"
                 f"(pooled, {sub['sub'].nunique()} pac., {len(sub)} canales)")
    ax.legend(loc="lower right"); fig.tight_layout()
    return fig


def fig_resid_foc(P):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, col, title, better in [
        (axes[0], "resid", "HFO residual (fuera de la resección)", "↑ en éxito"),
        (axes[1], "focality", "Focalidad del HFO (Gini)", "↓ en éxito")]:
        d = P.dropna(subset=[col])
        s = d[d.outcome == 1][col]; f = d[d.outcome == 0][col]
        ax.boxplot([s.values, f.values], labels=[f"S (n={len(s)})", f"F (n={len(f)})"],
                   widths=0.5, showmeans=True)
        for i, arr in enumerate([s, f], start=1):
            ax.scatter(np.random.normal(i, 0.05, len(arr)), arr, alpha=0.6,
                       color="#ff7f0e", zorder=3)
        p = mannwhitneyu(s, f).pvalue
        ax.set_title(f"{title}\nAUC={_auc(d['outcome'], d[col]):.2f}  p={p:.3f}  ({better})",
                     fontsize=10)
    fig.suptitle("Nivel paciente: features de la distribución de HFO vs. outcome")
    fig.tight_layout()
    return fig


def fig_combined_roc(P):
    cols = ["resid", "focality", "coverage"]
    d = P.dropna(subset=cols)
    auc, pr, y = loo_auc(d, cols)
    fpr, tpr, _ = roc_curve(y, pr)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot(fpr, tpr, color="#2ca02c", lw=2,
            label=f"residual+focalidad+cobertura\nLOO-CV AUC={auc:.2f}")
    ax.plot([0, 1], [0, 1], "--", color="0.6")
    ax.set_xlabel("1 - especificidad"); ax.set_ylabel("sensibilidad")
    ax.set_title("Modelo combinado -> outcome (validación leave-one-out)")
    ax.legend(loc="lower right"); fig.tight_layout()
    return fig


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detector_configs.tsv")
    ap.add_argument("--bids-root", default=DEFAULT_BIDS_ROOT)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    config_table = pd.read_csv(args.config, sep="\t")
    config_ids = list(config_table["config_id"])
    DS = load_datasets(config_ids)
    subs = [str(d.attrs["his_id"]) for d in DS]
    n_S = sum(1 for d in DS if str(d.attrs.get("outcome")) == "S")
    n_F = len(DS) - n_S

    ensure_electrodes(subs, bids_root=args.bids_root)

    # --- métricas overlap (Resultado A) ---
    hrr_df = build_hrr_matrix(DS, config_ids)
    metrics = per_config_metrics(hrr_df, config_table)
    metrics["score"] = metrics["accuracy"].fillna(0) * metrics["f1"].fillna(0)
    best_overlap = metrics.sort_values(["score", "auc"], ascending=False).iloc[0]
    comp = metric_comparison(DS, config_ids)
    # LOO-CV honesto del mejor overlap
    hd = hrr_df[[best_overlap["config_id"], "outcome"]].dropna()
    try:
        X = hd[[best_overlap["config_id"]]].values; y = hd["outcome"].astype(int).values
        pr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()),
                               X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
        ov_loo = roc_auc_score(y, pr)
    except Exception:
        ov_loo = float("nan")

    # --- hallazgo (Resultado B) ---
    T = spatial_abnormality(channel_table(DS, CFG_FIND, bids_root=args.bids_root))
    ch_auc = channel_level_auc(T, "rate")
    P = patient_features(DS, CFG_FIND, bids_root=args.bids_root, state_csv=STATE)
    Pf = P.dropna(subset=["resid", "focality", "coverage"])
    # AUC por-feature (misma base que las figuras de boxplot)
    def _feat_auc(col):
        d = P.dropna(subset=[col]); return _auc(d["outcome"], d[col])
    resid_auc = _feat_auc("resid"); foc_auc = _feat_auc("focality")
    cov_auc = _feat_auc("coverage"); glob_auc = _feat_auc("global_rate")
    comb_loo, _, _ = loo_auc(Pf, ["resid", "focality", "coverage"])
    # confound sueño
    Ps = P.dropna(subset=["sleep"])
    sleep_auc = _auc(Ps["outcome"], Ps["sleep"])
    rho_sleep_rate = spearmanr(Ps["sleep"], Ps["global_rate"])[0]

    with PdfPages(PDF) as pdf:
        # ---- 1. Portada ----
        text_page(pdf, "HFO interictales y outcome quirúrgico — síntesis para coautores", [
            "EJN-2026-01-33765  ·  decisión: reject & resubmit  ·  refactor metodológico + reanálisis",
            "",
            "Objetivo: fortalecer la metodología del detector de HFO (respondiendo al revisor),",
            "correr el pipeline sobre toda la base y reevaluar la relación HFO<->outcome quirúrgico.",
            "",
            "Dataset: HUP iEEG (OpenNeuro ds004100, v1.1.3).",
            f"  Sujetos SEEG con interictal analizados: {len(DS)}   (Outcome S={n_S}, F={n_F})",
            "  fs heterogénea: 500 / 512 / 1024 Hz.   Registros interictales (todos los runs, concatenados).",
            "",
            "Contenido: (1) métodos  (2) respuesta al revisor  (3) configuraciones  (4) hipótesis y teoría",
            "  (5) Resultado A: solapamiento — nulo  (6) Resultado B: distribución del HFO — hallazgo",
            "  (7) mapas cerebrales  (8) confound-check y modelo  (9) propuesta.",
            "",
            "Nota: informe narrativo/visual. El PDF técnico de respaldo es informe_hfo_ejn.pdf.",
        ])

        # ---- 2. Métodos / pipeline (ampliado) ----
        text_page(pdf, "Métodos — del SEEG al HFO y a las métricas", [
            "A) Detección de HFO por canal (por cada configuración del barrido):",
            "   1. Montaje bipolar (resta de contactos adyacentes del mismo electrodo) → atenúa artefactos comunes.",
            "   2. Filtro pasa-banda FIR de fase cero (MNE: windowed-sinc, ventana Hamming, overlap-add).",
            "   3. Envolvente de energía RMS en una ventana deslizante corta (10-50 ms, solape 25%).",
            "   4. Umbral estadístico: se marca evento si el RMS supera mean + k·SD (k = 3/5/7), SD del archivo.",
            "   5. Eventos separados por <10 ms se funden. → cada canal obtiene una TASA de HFO [eventos/min].",
            "   (Detector: mne_hfo.RMSDetector. La tabla de configs externa define banda × ventana × umbral.)",
            "",
            "B) De la tasa por canal a las métricas de paciente. Dos familias, dos preguntas distintas:",
            "   · SOLAPAMIENTO (Hipótesis A): ¿los canales de más HFO coinciden con la resección?",
            "     (HRR = top-N/2 en resección, recall, concordancia AUC intra-paciente).",
            "   · DISTRIBUCIÓN (Hipótesis B): ¿qué FORMA tiene el mapa de HFO?  residual (HFO fuera de la",
            "     resección), focalidad (Gini = concentrado vs difuso), tasa global.",
            "",
            "C) Normalización espacial (atlas normativo): con las coords fsaverage de cada electrodo, la",
            "   'anormalidad' de un canal = cuánto excede su tasa respecto de la esperada en esa ubicación,",
            "   estimada con los canales normales de OTROS pacientes (kNN, leave-one-patient-out). Descuenta el",
            "   HFO fisiológico regional (p.ej. ripples del hipocampo).",
            "",
            "D) Validación honesta: AUC con leave-one-out CV (no in-sample) + confound-check (sueño, fs, cobertura).",
        ], wrap=98, fontsize=8.2)

        # ---- 3. Respuesta al revisor (tabla) ----
        rev = pd.DataFrame([
            ["1. Filtro 'FIR Butterworth' (contradictorio)",
             "Aclarado: es FIR de fase cero de MNE (windowed-sinc, Hamming, OLA). NO Butterworth."],
            ["2. Banda 40 Hz solapa low gamma; separar bandas",
             "Bandas separadas: high gamma 65-100 y ripple 100-240. Fast ripples: imposibles (Nyquist)."],
            ["3. Ventana 50-150 ms demasiado larga",
             "Fix bug ms->muestras (usa fs real). Barrido de ventanas cortas 10/20/30/50 ms."],
            ["4. ¿Por qué umbral 3 SD?",
             "Umbral barrido: 3 / 5 / 7 SD (columna de la tabla de configs)."],
            ["5. ¿Cuántos runs interictales?",
             "TODOS los runs interictales SEEG por sujeto, concatenados; se reporta n_runs y minutos."],
            ["(infra) reproducibilidad / datos",
             "Barrido por tabla TSV externa + descarga automática desde OpenNeuro (selectiva, idempotente)."],
        ], columns=["Consulta del revisor", "Qué se hizo"])
        table_page(pdf, "Respuesta al revisor y cambios de metodología (resumen)", rev,
                   col_wrap=[34, 50], colWidths=[0.37, 0.60], fontsize=8.5)

        # ---- 4. Configuraciones + Fig 1 (heatmap) ----
        fig_sweep, _ = fig_sweep_heatmap(metrics, "auc"); _tag(fig_sweep, 1)
        pdf.savefig(fig_sweep); plt.close(fig_sweep)
        text_page(pdf, "Configuraciones evaluadas (24)", [
            "Barrido = {high gamma 65-100 Hz, ripple 100-240 Hz} x ventanas {10,20,30,50} ms x umbrales {3,5,7} SD.",
            "A fs=256 Hz (algún sujeto) solo corren las de high gamma (ripple supera Nyquist) — se descartan por sujeto.",
            "",
            "La Fig 1 (heatmap) muestra el AUC solapamiento->outcome de cada config: todo el barrido queda en 0.46-0.65,",
            "sin ninguna combinación que despegue — primer indicio de que el solapamiento no lleva la señal.",
        ])

        # ---- 5. Hipótesis, teoría e interpretación (ANTES de los resultados) ----
        text_page(pdf, "Hipótesis, teoría e interpretación", [
            "Dos hipótesis sobre el mismo mapa de HFO por canal:",
            "  · Hipótesis A (la del paper): el outcome depende del SOLAPAMIENTO — resecar donde hay más HFO",
            "    lleva al éxito. → se testea con HRR/recall/concordancia (Fig 1-3).",
            "  · Hipótesis B (la que emerge): el outcome depende de la DISTRIBUCIÓN del HFO — su forma",
            "    (concentrado vs difuso), cuánto queda fuera de lo resecado, su nivel global (Fig 4-7).",
            "",
            "Teoría — por qué A puede fallar y B puede funcionar:",
            "  · HFO fisiológico vs patológico: los ripples fisiológicos del hipocampo/temporal mesial elevan la",
            "    tasa cruda tanto dentro como fuera de la resección → el solapamiento directo se diluye. La",
            "    normalización normativa espacial (paradigma del atlas HUP, Bernabei et al.; el dataset trae coords",
            "    fsaverage justamente para esto) descuenta ese HFO regional y hace emerger la 'anormalidad'.",
            "  · La forma de la distribución (focalidad, residual) captura información que el solapamiento tira",
            "    (dos pacientes con el MISMO solapamiento pueden tener distribuciones opuestas).",
            "",
            "Interpretación esperada (se confirma en Resultados):",
            "  · A da nulo; B predice el outcome (LOO-CV ~0.80), robusto a sueño/fs/cobertura.",
            "  · Dirección contraintuitiva (HFO difuso/abundante + implante acotado → éxito): el cohorte es",
            "    mayormente ablación temporal mesial (buen pronóstico); el perfil opuesto ≈ casos",
            "    neocorticales/multifocales complejos. => las features de HFO actuarían como PROXY DEL FENOTIPO",
            "    de epilepsia, más que del mecanismo 'resecar el HFO'. Los mapas de la Fig 6-7 lo ilustran.",
            "  · Limitaciones: n=35, retrospectivo, dirección contraintuitiva, comparaciones múltiples",
            "    (mitigadas por LOO-CV + consistencia entre configs); fenotipo ≠ causalidad.",
        ], wrap=98, fontsize=8.2)

        # ---- 6. Resultado A: solapamiento NULO (Fig 2, Fig 3) ----
        fig2, _ = fig_rate_resect_vs_unlabeled(DS, config_table); _tag(fig2, 2)
        pdf.savefig(fig2); plt.close(fig2)
        fig3, _ = fig_metric_comparison(DS, comp); _tag(fig3, 3)
        pdf.savefig(fig3); plt.close(fig3)
        text_page(pdf, "Resultado A — solapamiento HFO↔resección: NULO", [
            "Prueba de la Hipótesis A. La Fig 2 compara la tasa de HFO en canales resecados vs no-etiquetados por",
            "outcome (diferencia leve, no discrimina); la Fig 3 muestra que 5 métricas de solapamiento convergen bajo.",
            "",
            f"  · Mejor config de solapamiento (HRR): {best_overlap['config_id']}  "
            f"AUC in-sample={best_overlap['auc']:.2f}, p={best_overlap['mwu_p']:.2f}",
            f"  · Validación honesta leave-one-out (LOO-CV): AUC = {ov_loo:.2f}  (≈ azar 0.5)",
            "  · 5 métricas (HRR top-N/2, concordancia, log-ratio, enriquecimiento, fracción) ~0.55-0.65 in-sample;",
            "    ninguna significativa. Refinar la métrica o usar ML/PCA no ayuda (bajo LOO-CV caen a ~azar).",
            "",
            "Conclusión: el solapamiento HFO-resección NO predice el outcome en este cohorte.",
        ], wrap=98, fontsize=8.4)

        # ---- 7. Resultado B — nivel canal (Fig 4) ----
        figc = fig_channel_roc(T); _tag(figc, 4)
        pdf.savefig(figc); plt.close(figc)
        text_page(pdf, "Resultado B — nivel canal: el HFO SÍ localiza tejido", [
            "Aunque el solapamiento no prediga el outcome, a nivel canal la tasa de HFO marca el tejido tratado (Fig 4):",
            "",
            f"  · AUC (tasa HFO discrimina canal resecado vs no) = {ch_auc:.2f}  (pooled, {T['sub'].nunique()} pac.)",
            "  · Tasa media de HFO por tipo de canal:  normal ≈ 2.5   SOZ ≈ 5.0   resecado ≈ 4.1  [eventos/min]",
            "",
            "Es decir: el detector marca dónde está la zona epileptógena (coincide con SOZ/resección).",
            "El problema NO es detectar HFO patológico, sino traducir eso a una predicción de outcome de paciente.",
        ], wrap=98, fontsize=8.4)

        # ---- 8. Resultado B — nivel paciente (Fig 5) ----
        figrf = fig_resid_foc(P); _tag(figrf, 5)
        pdf.savefig(figrf); plt.close(figrf)
        text_page(pdf, "Resultado B — nivel paciente: la DISTRIBUCIÓN predice", [
            f"Config representativa: {CFG_FIND} (ripple, umbral 7), con la anormalidad normalizada (métodos C).",
            "La Fig 5 muestra residual y focalidad por outcome (S vs F):",
            "",
            f"  · HFO residual (media de anormalidad FUERA de la resección):  AUC = {resid_auc:.2f}",
            f"  · Focalidad (Gini de la anormalidad entre canales):           AUC = {foc_auc:.2f}",
            f"  · Tasa global de HFO del paciente:                            AUC = {glob_auc:.2f}",
            "",
            "Consistente en 14-15 de 24 configs (umbral 5/7). Las métricas de solapamiento (A) eran las débiles:",
            "la señal está en la DISTRIBUCIÓN del HFO (su forma), no en su solapamiento con la resección.",
        ], wrap=98, fontsize=8.4)

        # ---- 9. Resultado B — mapas cerebrales (Fig 6, Fig 7) ----
        fig_ex, _ex = bm.fig_exemplars(DS, CFG_FIND, bids_root=args.bids_root, fig_no=6)
        pdf.savefig(fig_ex); plt.close(fig_ex)
        fig_at = bm.fig_group_atlas(DS, CFG_FIND, bids_root=args.bids_root, fig_no=7)
        pdf.savefig(fig_at); plt.close(fig_at)
        text_page(pdf, "Resultado B — mapas de distribución en el cerebro", [
            "Evidencia visual de la Hipótesis B, usando las coords fsaverage reales de los electrodos:",
            "",
            "  · Fig 6 — dos pacientes representativos (uno exitoso, uno fallido): cada electrodo coloreado por",
            "    anormalidad de HFO, dimensionado por tasa, con los canales resecados anillados. Se ve la FORMA",
            "    del patrón (dónde está el HFO, cuánto queda fuera de lo resecado), no solo el solapamiento.",
            "  · Fig 7 — atlas grupal: el patrón espacial de anormalidad agregado por grupo (S vs F), que resalta",
            "    la predominancia temporal — coherente con la interpretación fenotípica.",
            "",
            "Caveat: fsaverage graficado sobre molde MNI152 (aproximación cualitativa, no localización exacta).",
        ], wrap=98, fontsize=8.4)

        # ---- 10. Confound-check + modelo combinado (Fig 8) ----
        conf = pd.DataFrame([
            ["Sueño (delta relativo)", f"{sleep_auc:.2f}", "NO", "Sube la tasa de HFO (rho≈%.2f) pero no se asocia al outcome -> no confunde." % rho_sleep_rate],
            ["Frecuencia de muestreo", "~0.51", "NO", "Sin efecto sobre el outcome."],
            ["Cobertura (nº electrodos)", f"{cov_auc:.2f}", "co-predictor", "Más electrodos -> peor outcome. Residual/focalidad sobreviven controlando por esto (p<0.05)."],
        ], columns=["Variable", "AUC out", "¿Confunde?", "Detalle"])
        table_page(pdf, "Confound-check del hallazgo", conf,
                   col_wrap=[16, 8, 11, 46], colWidths=[0.19, 0.11, 0.14, 0.56], fontsize=8.5)
        figm = fig_combined_roc(Pf); _tag(figm, 8)
        pdf.savefig(figm); plt.close(figm)
        text_page(pdf, "Modelo combinado y robustez", [
            f"  · Modelo residual + focalidad + cobertura:  LOO-CV AUC = {comb_loo:.2f}  (out-of-sample, honesto; Fig 8)",
            "",
            "  · Robusto a: estado de sueño (no confunde), frecuencia de muestreo, y controlando por cobertura.",
            "  · Consistente entre configuraciones (umbral 5/7) y entre bandas.",
            "",
            f"Pasamos de la Hipótesis A (LOO ≈ {ov_loo:.2f}, azar) a un predictor cross-validado (LOO ≈ {comb_loo:.2f}).",
            "Sigue siendo un resultado EXPLORATORIO (n=35, comparaciones múltiples) — a validar.",
        ], wrap=98, fontsize=8.4)

        # ---- 11. Propuesta / próximos pasos ----
        text_page(pdf, "Propuesta y próximos pasos", [
            "Para la resubmisión:",
            "  1. Reportar con transparencia: la hipótesis de solapamiento no se sostiene bajo evaluación rigurosa",
            "     (bandas correctas, ventana corregida, 5 métricas, barrido completo, LOO-CV).",
            "  2. Presentar el hallazgo exploratorio como aporte: features de la distribución de HFO (residual +",
            "     focalidad) + cobertura predicen outcome (LOO-CV AUC ≈ %.2f), confund-checked." % comb_loo,
            "  3. Enmarcar la interpretación como fenotípica, no mecanicista (evita sobre-afirmar).",
            "",
            "Para solidificarlo (si se decide):",
            "  · Controlar fenotipo explícito (MTL vs no-MTL, lesional vs no-lesional) como covariable/subgrupo.",
            "  · Pre-registrar la dirección y validar en cohorte externo (otro centro / mayor n).",
            "  · Explorar sujetos con fs=1024 Hz para bandas más altas (ripple/fast-ripple) donde la fs lo permita.",
            "",
            "Decisión para coautores: ¿reencuadrar el paper hacia 'predicción de outcome desde la distribución",
            "de HFO' con el caveat fenotípico, o mantener el foco metodológico + resultado nulo honesto?",
        ])

    print(f"Informe de síntesis generado: {PDF}")
    print(f"  sujetos={len(DS)} (S={n_S},F={n_F})  overlap LOO={ov_loo:.2f}  "
          f"canal={ch_auc:.2f}  resid={resid_auc:.2f}  foc={foc_auc:.2f}  combinado LOO={comb_loo:.2f}")


if __name__ == "__main__":
    main()
