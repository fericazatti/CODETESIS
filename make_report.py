#!/usr/bin/env python3
"""
Genera el informe PDF de resultados de la corrida completa de HFO.

Lee todos los pickles de data/processed/ que correspondan al barrido actual
(config_ids de configs/detector_configs.tsv), calcula HRR por (sujeto x config),
ajusta regresión logística HRR->outcome por config (accuracy/f1/AUC) y el test
de Mann-Whitney (S vs F), y arma figuras estilo paper + tablas en un único PDF:

    data/results/informe_hfo_ejn.pdf

Uso:  python make_report.py [--config configs/detector_configs.tsv]
"""
import argparse
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit

PROCESSED_DIR = "data/processed"
RESULTS_DIR = "data/results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
PDF_PATH = os.path.join(RESULTS_DIR, "informe_hfo_ejn.pdf")

RESECT = ["resect", "resect,soz"]


# ----------------------------------------------------------------------------- carga
def load_datasets(config_ids):
    """Carga los pickles cuyos data_vars coinciden con el barrido actual."""
    cfg_set = set(config_ids)
    datasets = []
    for path in sorted(glob.glob(os.path.join(PROCESSED_DIR, "sub-*_processed.pickle"))):
        with open(path, "rb") as f:
            ds = pickle.load(f)
        # incluye el sujeto si tiene AL MENOS una config del barrido actual
        # (cohorte con fs heterogénea: a fs=256 solo hay configs high gamma).
        if cfg_set & set(map(str, ds.data_vars)):
            datasets.append(ds)
    return datasets


def outcome_bin(ds):
    return 1 if str(ds.attrs.get("outcome")) == "S" else 0


# ----------------------------------------------------------------------------- HRR
def hrr_for(ds, cfg):
    """HRR de un sujeto para una config (patrón de analysis2.py). NaN si no aplica."""
    df = ds[cfg].to_pandas().T.reset_index()
    cond = df["status"].isin(RESECT)
    resection_size = int(cond.sum())
    if resection_size == 0:
        return np.nan
    resection_size = round(resection_size / 2)
    if resection_size == 0:
        return np.nan
    hfo_region = df.sort_values("counts", ascending=False).head(resection_size)
    hfo_in = int(hfo_region["status"].isin(RESECT).sum())
    return round(hfo_in / resection_size, 3)


def build_hrr_matrix(datasets, config_ids):
    """DataFrame index=sujeto, columnas=config_ids (+ 'outcome')."""
    rows = {}
    for ds in datasets:
        sid = str(ds.attrs["his_id"])
        present = set(map(str, ds.data_vars))
        rows[sid] = {cfg: (hrr_for(ds, cfg) if cfg in present else np.nan)
                     for cfg in config_ids}
        rows[sid]["outcome"] = outcome_bin(ds)
    return pd.DataFrame(rows).T


# --------------------------------------------------------------- métricas alternativas
# Se probaron varias formas de cuantificar el solapamiento HFO<->resección, para
# ver si el HRR original (top N/2, binario/cuantizado) subestimaba la señal.
def _pc(ds, cfg):
    df = ds[cfg].to_pandas().T.reset_index()
    df["y"] = df["status"].isin(RESECT).astype(int)
    df["rate"] = pd.to_numeric(df["hfo_rate"], errors="coerce").fillna(0.0)
    df["counts"] = pd.to_numeric(df["counts"], errors="coerce").fillna(0.0)
    return df


def m_hrr(ds, cfg):
    return hrr_for(ds, cfg)


def m_concordance(ds, cfg):
    """AUC intra-paciente del HFO rate separando canales resect vs no-resect."""
    df = _pc(ds, cfg)
    if df["y"].nunique() < 2:
        return np.nan
    from sklearn.metrics import roc_auc_score as _auc
    return _auc(df["y"], df["rate"])


def m_enrichment(ds, cfg):
    """Enriquecimiento: fracción de carga HFO en resección / fracción de canales."""
    df = _pc(ds, cfg)
    tot = df["counts"].sum()
    nres = int(df["y"].sum())
    if tot <= 0 or nres == 0:
        return np.nan
    frac_hfo = df.loc[df["y"] == 1, "counts"].sum() / tot
    return frac_hfo / (nres / len(df))


METRICS = {
    "HRR (top N/2)": m_hrr,
    "Concordancia AUC": m_concordance,
    "Enriquecimiento": m_enrichment,
}


def metric_series(datasets, cfg, metric_fn):
    """(valores, outcomes) de una métrica para una config, sobre los sujetos válidos."""
    vals, ys = [], []
    for ds in datasets:
        if cfg not in set(map(str, ds.data_vars)):
            continue
        v = metric_fn(ds, cfg)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        vals.append(float(v))
        ys.append(outcome_bin(ds))
    return np.array(vals), np.array(ys)


def metric_comparison(datasets, config_ids):
    """Para cada métrica, la mejor config por AUC(outcome), con n y p (Mann-Whitney)."""
    out = []
    for name, fn in METRICS.items():
        best = None
        for cfg in config_ids:
            vals, ys = metric_series(datasets, cfg, fn)
            if len(vals) < 8 or len(np.unique(ys)) < 2:
                continue
            auc = roc_auc_score(ys, vals)
            auc = max(auc, 1 - auc)
            p = mannwhitneyu(vals[ys == 1], vals[ys == 0]).pvalue
            if best is None or auc > best["auc"]:
                best = {"metric": name, "best_config": cfg, "n": len(vals),
                        "auc": round(auc, 3), "p": round(float(p), 4)}
        if best:
            out.append(best)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- métricas por config
def per_config_metrics(hrr_df, config_table):
    out = []
    for _, c in config_table.iterrows():
        cfg = c["config_id"]
        sub = hrr_df[[cfg, "outcome"]].dropna()
        x = sub[cfg].astype(float).values
        y = sub["outcome"].astype(int).values
        rec = {
            "config_id": cfg, "band_label": c["band_label"],
            "band_hz": f"{int(c['f_low'])}-{int(c['f_high'])}",
            "window_ms": int(c["window_ms"]), "threshold_sd": int(c["threshold_sd"]),
            "n": len(y), "auc": np.nan, "f1": np.nan, "accuracy": np.nan, "mwu_p": np.nan,
        }
        if len(np.unique(y)) == 2 and len(y) >= 4:
            try:
                model = LogisticRegression().fit(x.reshape(-1, 1), y)
                yhat = model.predict(x.reshape(-1, 1))
                proba = model.predict_proba(x.reshape(-1, 1))[:, 1]
                rec["accuracy"] = float((yhat == y).mean())
                rec["f1"] = float(f1_score(y, yhat, zero_division=0))
                rec["auc"] = float(roc_auc_score(y, proba))
            except Exception:
                pass
            try:
                s = x[y == 1]; fk = x[y == 0]
                if len(s) and len(fk):
                    rec["mwu_p"] = float(mannwhitneyu(s, fk, alternative="two-sided").pvalue)
            except Exception:
                pass
        out.append(rec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figuras
def fig_rate_resect_vs_unlabeled(datasets, config_table):
    """F1 estilo Figura 6: HFO rate medio resect vs unlabeled, por outcome y banda."""
    bands = list(dict.fromkeys(config_table["band_label"]))
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4.2), squeeze=False)
    for ax, band in zip(axes[0], bands):
        # config representativa de la banda: umbral mediano, ventana mediana
        cand = config_table[config_table["band_label"] == band].sort_values(
            ["threshold_sd", "window_ms"])
        cfg = cand.iloc[len(cand) // 2]["config_id"]
        data = {"S": {"resect": [], "unlabeled": []}, "F": {"resect": [], "unlabeled": []}}
        for ds in datasets:
            if cfg not in set(map(str, ds.data_vars)):
                continue  # sujeto sin esta banda (p.ej. fs=256 y banda ripple)
            grp = "S" if outcome_bin(ds) == 1 else "F"
            df = ds[cfg].to_pandas().T.reset_index()
            r = df.loc[df["status"].isin(RESECT), "hfo_rate"].astype(float)
            u = df.loc[~df["status"].isin(RESECT), "hfo_rate"].astype(float)
            if len(r):
                data[grp]["resect"].append(r.mean())
            if len(u):
                data[grp]["unlabeled"].append(u.mean())
        labels = ["Outcome S", "Outcome F"]
        resect_means = [np.nanmean(data["S"]["resect"] or [np.nan]),
                        np.nanmean(data["F"]["resect"] or [np.nan])]
        unlab_means = [np.nanmean(data["S"]["unlabeled"] or [np.nan]),
                       np.nanmean(data["F"]["unlabeled"] or [np.nan])]
        resect_sem = [np.nanstd(data["S"]["resect"] or [np.nan]),
                      np.nanstd(data["F"]["resect"] or [np.nan])]
        unlab_sem = [np.nanstd(data["S"]["unlabeled"] or [np.nan]),
                     np.nanstd(data["F"]["unlabeled"] or [np.nan])]
        xpos = np.arange(2); w = 0.35
        ax.bar(xpos - w / 2, resect_means, w, yerr=resect_sem, capsize=4,
               color="#2ca02c", label="resect")
        ax.bar(xpos + w / 2, unlab_means, w, yerr=unlab_sem, capsize=4,
               color="#4e4e4e", label="unlabeled")
        ax.set_xticks(xpos); ax.set_xticklabels(labels)
        ax.set_ylabel("Mean HFO rate [counts/min]")
        ax.set_title(f"{band}  ({cfg})")
        ax.legend()
    fig.suptitle("HFO rate: resected vs unlabeled channels, by surgical outcome")
    fig.tight_layout()
    return _save(fig, "F1_rate_resect_vs_unlabeled.png")


def fig_hrr_by_outcome(hrr_df, best_cfg, mwu_p):
    """F2: distribución de HRR S vs F para la mejor config."""
    sub = hrr_df[[best_cfg, "outcome"]].dropna()
    s = sub.loc[sub["outcome"] == 1, best_cfg].astype(float)
    fk = sub.loc[sub["outcome"] == 0, best_cfg].astype(float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.boxplot([s.values, fk.values], labels=["S (n=%d)" % len(s), "F (n=%d)" % len(fk)],
               widths=0.5, showmeans=True)
    for i, arr in enumerate([s, fk], start=1):
        ax.scatter(np.random.normal(i, 0.05, len(arr)), arr, alpha=0.6,
                   color="#1f77b4", zorder=3)
    ax.set_ylabel("HRR")
    ptxt = f"Mann-Whitney p = {mwu_p:.3f}" if np.isfinite(mwu_p) else "Mann-Whitney p = n/a"
    ax.set_title(f"HRR by outcome — {best_cfg}\n{ptxt}")
    fig.tight_layout()
    return _save(fig, "F2_hrr_by_outcome.png")


def fig_sweep_heatmap(metrics, value="auc"):
    """F3: heatmap del barrido banda×ventana (filas) × umbral (cols) del valor dado."""
    m = metrics.copy()
    m["row"] = m["band_label"] + " " + m["band_hz"] + "  w" + m["window_ms"].astype(str) + "ms"
    piv = m.pivot_table(index="row", columns="threshold_sd", values=value)
    piv = piv.sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(piv) + 1.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c} SD" for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.7 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label=value.upper())
    ax.set_title(f"Barrido de configuraciones — {value.upper()} de HRR->outcome")
    fig.tight_layout()
    return _save(fig, f"F3_sweep_{value}.png")


def fig_logreg(hrr_df, best_cfg):
    """F4: curva de regresión logística HRR->outcome de la mejor config."""
    sub = hrr_df[[best_cfg, "outcome"]].dropna()
    x = sub[best_cfg].astype(float).values; y = sub["outcome"].astype(int).values
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y, color="#1f77b4", zorder=20, label="sujetos")
    if len(np.unique(y)) == 2:
        model = LogisticRegression().fit(x.reshape(-1, 1), y)
        xs = np.linspace(min(x) - 0.05, max(x) + 0.05, 300)
        ys = expit(xs * model.coef_[0][0] + model.intercept_[0])
        ax.plot(xs, ys, color="red", lw=2, label="regresión logística")
    ax.axhline(0.5, color="0.6", ls="--")
    ax.set_xlabel("HRR"); ax.set_ylabel("outcome (1=S, 0=F)")
    ax.set_ylim(-0.15, 1.15); ax.legend()
    ax.set_title(f"HRR vs outcome — {best_cfg}")
    fig.tight_layout()
    return _save(fig, "F4_logreg.png")


def fig_channel_bar(datasets, best_cfg):
    """F5: HFO rate por canal de un sujeto representativo (estilo hfo_plots.bar_chart)."""
    # un sujeto con outcome S que TENGA la mejor config
    cands = [d for d in datasets
             if best_cfg in set(map(str, d.data_vars)) and outcome_bin(d) == 1]
    if not cands:
        cands = [d for d in datasets if best_cfg in set(map(str, d.data_vars))]
    ds = cands[0]
    df = ds[best_cfg].to_pandas().T.reset_index()
    df = df.dropna(subset=["hfo_rate"]).sort_values("hfo_rate", ascending=False).head(40)
    colors = ["#2ca02c" if s in RESECT else "#4e4e4e" for s in df["status"]]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(df["channels"].astype(str), df["hfo_rate"].astype(float), color=colors, width=0.7)
    p75 = np.percentile(df["hfo_rate"].astype(float), 75)
    ax.axhline(p75, color="red", ls="--", label="p75")
    ax.set_ylabel("HFO rate [counts/min]"); ax.set_xlabel("Canales (top 40)")
    ax.set_title(f"HFO por canal — {ds.attrs['his_id']} (outcome "
                 f"{ds.attrs.get('outcome')}) — {best_cfg}\nverde=resect, gris=unlabeled")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.legend(); fig.tight_layout()
    return _save(fig, "F5_channel_bar.png")


def fig_metric_comparison(datasets, comp):
    """F6: (izq) AUC-outcome de la mejor config por métrica; (der) concordancia S vs F."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # panel izquierdo: barras de AUC(outcome) por métrica
    axL.bar(comp["metric"], comp["auc"], color="#1f77b4")
    axL.axhline(0.5, color="0.5", ls="--", label="azar (0.5)")
    for i, (a, p) in enumerate(zip(comp["auc"], comp["p"])):
        axL.text(i, a + 0.01, f"AUC={a:.2f}\np={p:.2f}", ha="center", va="bottom", fontsize=8)
    axL.set_ylim(0, 0.9); axL.set_ylabel("AUC métrica -> outcome")
    axL.set_title("Robustez: mejor config por métrica")
    axL.tick_params(axis="x", labelsize=8); axL.legend(fontsize=8)

    # panel derecho: concordancia intra-paciente S vs F (su mejor config)
    conc_row = comp[comp["metric"] == "Concordancia AUC"]
    if len(conc_row):
        cfg = conc_row.iloc[0]["best_config"]
        vals, ys = metric_series(datasets, cfg, m_concordance)
        s = vals[ys == 1]; fk = vals[ys == 0]
        axR.boxplot([s, fk], labels=[f"S (n={len(s)})", f"F (n={len(fk)})"],
                    widths=0.5, showmeans=True)
        for i, arr in enumerate([s, fk], start=1):
            axR.scatter(np.random.normal(i, 0.05, len(arr)), arr, alpha=0.6,
                        color="#ff7f0e", zorder=3)
        axR.axhline(0.5, color="0.6", ls="--")
        p = mannwhitneyu(s, fk).pvalue if len(s) and len(fk) else np.nan
        axR.set_ylabel("Concordancia intra-paciente (AUC canal)")
        axR.set_title(f"Concordancia por outcome — {cfg}\nMann-Whitney p={p:.3f}")
    fig.tight_layout()
    return _save(fig, "F6_metric_comparison.png")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150)
    return fig, path


# ----------------------------------------------------------------------------- páginas de texto/tabla
def text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.08, 0.94, title, fontsize=16, weight="bold", va="top")
    fig.text(0.08, 0.88, "\n".join(lines), fontsize=9.5, va="top", family="monospace")
    pdf.savefig(fig); plt.close(fig)


def table_page(pdf, title, df, fontsize=7):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off"); ax.set_title(title, fontsize=13, weight="bold", loc="left", pad=20)
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="upper center",
                 cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 1.25)
    pdf.savefig(fig); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detector_configs.tsv")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config_table = pd.read_csv(args.config, sep="\t")
    config_ids = list(config_table["config_id"])

    datasets = load_datasets(config_ids)
    if not datasets:
        raise SystemExit("No hay pickles del barrido actual en data/processed/. Corré run_full.py primero.")

    n_S = sum(outcome_bin(d) == 1 for d in datasets)
    n_F = len(datasets) - n_S

    hrr_df = build_hrr_matrix(datasets, config_ids)
    metrics = per_config_metrics(hrr_df, config_table)

    # ranking por acc*f1 (con AUC de desempate)
    metrics["score"] = metrics["accuracy"].fillna(0) * metrics["f1"].fillna(0)
    metrics_sorted = metrics.sort_values(["score", "auc"], ascending=False)
    best_cfg = metrics_sorted.iloc[0]["config_id"]
    best_p = metrics_sorted.iloc[0]["mwu_p"]

    # comparación de métricas de solapamiento (robustez)
    comp = metric_comparison(datasets, config_ids)

    # cohorte y runs
    from src.utils.analysis_utils import organize_results
    cohort = organize_results(datasets)
    runs_csv = os.path.join(RESULTS_DIR, "interictal_runs_report.csv")
    runs_df = pd.read_csv(runs_csv) if os.path.exists(runs_csv) else pd.DataFrame()

    # figuras
    figs = [
        fig_rate_resect_vs_unlabeled(datasets, config_table),
        fig_hrr_by_outcome(hrr_df, best_cfg, best_p),
        fig_sweep_heatmap(metrics, "auc"),
        fig_logreg(hrr_df, best_cfg),
        fig_channel_bar(datasets, best_cfg),
        fig_metric_comparison(datasets, comp),
    ]

    # PDF
    with PdfPages(PDF_PATH) as pdf:
        text_page(pdf, "Detección de HFO interictales vs. resultado quirúrgico", [
            "Informe de resultados — refactor metodológico (revisión EJN-2026-01-33765)",
            "Dataset: HUP iEEG (OpenNeuro ds004100), registros interictales SEEG.",
            "",
            f"Sujetos analizados: {len(datasets)}   (Outcome S={n_S}, F={n_F})",
            f"Configuraciones barridas: {len(config_ids)}",
            f"Bandas: {sorted(set(config_table['band_label']))}",
            f"Ventanas [ms]: {sorted(set(config_table['window_ms']))}   "
            f"Umbrales [SD]: {sorted(set(config_table['threshold_sd']))}",
            "",
            "Método (responde a la revisión):",
            " 1) Filtro FIR de fase cero de MNE (windowed-sinc, Hamming, overlap-add).",
            "    (El texto original decía 'FIR Butterworth', corregido.)",
            " 2) Bandas separadas high gamma / ripple. Fast ripples NO analizables:",
            "    fs=500-512 Hz -> Nyquist ~250-256 Hz.",
            " 3) Ventana RMS corta (10-30 ms; a fs~500 la mínima representable ~10 ms).",
            " 4) Umbral barrido en 3/5/7 SD.",
            " 5) Se concatenan TODOS los runs interictales por sujeto (ver tabla de runs).",
            "",
            f"Mejor configuración (acc*f1): {best_cfg}",
            f"  AUC={metrics_sorted.iloc[0]['auc']:.3f}  "
            f"acc={metrics_sorted.iloc[0]['accuracy']:.3f}  "
            f"f1={metrics_sorted.iloc[0]['f1']:.3f}  "
            f"MannWhitney p={best_p:.3f}" if np.isfinite(best_p) else
            f"  AUC={metrics_sorted.iloc[0]['auc']:.3f}",
        ])
        for fig, _ in figs:
            pdf.savefig(fig); plt.close(fig)

        table_page(pdf, "Robustez: comparación de métricas de solapamiento HFO-resección",
                   comp.round(3))

        top = metrics_sorted.head(12)[
            ["config_id", "band_label", "band_hz", "window_ms", "threshold_sd",
             "n", "accuracy", "f1", "auc", "mwu_p"]].round(3)
        table_page(pdf, "Ranking de configuraciones (top 12)", top)

        cohort_show = cohort.reset_index()[
            ["id", "outcome", "engel", "therapy", "implant", "target", "lesion_status"]]
        table_page(pdf, f"Cohorte ({len(cohort_show)} sujetos)", cohort_show, fontsize=6)

        if not runs_df.empty:
            table_page(pdf, "Runs interictales por sujeto", runs_df.round(2), fontsize=6)

    print(f"Informe generado: {PDF_PATH}")
    print(f"Figuras: {FIG_DIR}")
    print(f"Sujetos={len(datasets)} (S={n_S},F={n_F})  mejor={best_cfg}")


if __name__ == "__main__":
    main()
