"""
Mapas de distribución de HFO en el espacio del cerebro (glass brain, nilearn).

Usa las coordenadas fsaverage reales de los electrodos (advanced_metrics.channel_table)
y la anormalidad espacial (spatial_abnormality) para mostrar DÓNDE está el HFO y con
qué forma se distribuye — evidencia visual de la Hipótesis B.

Caveat: las coords están en espacio fsaverage; se grafican sobre el molde MNI152 de
nilearn (aproximación válida para un mapa cualitativo, no para localización exacta).
"""
import os
import tempfile
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.utils.advanced_metrics import (channel_table, spatial_abnormality, _gini,
                                        RESECT)
from src.utils.cloud_data import DEFAULT_BIDS_ROOT

FSAVERAGE_CAVEAT = ("Electrodos en espacio fsaverage graficados sobre molde MNI152 "
                    "(aprox. cualitativa). Color = anormalidad de HFO; tamaño = tasa; "
                    "aro azul = canal resecado.")


def _abn_table(datasets, cfg, bids_root=DEFAULT_BIDS_ROOT):
    return spatial_abnormality(channel_table(datasets, cfg, bids_root=bids_root))


def select_exemplars(datasets, cfg, bids_root=DEFAULT_BIDS_ROOT, min_cov=40):
    """
    Elige 1 sujeto S y 1 F REPRESENTATIVOS: los más cercanos al centroide de su grupo
    en el espacio (residual, focalidad) estandarizado, con cobertura >= min_cov canales.
    Devuelve (sub_S, sub_F, feats_df).
    """
    T = _abn_table(datasets, cfg, bids_root)
    rows = []
    for s, g in T.groupby("sub"):
        g = g.dropna(subset=["abn"])
        if g["resect"].sum() == 0 or len(g) < min_cov:
            continue
        non = g[g["resect"] == 0]
        outc = int(g["outcome"].iloc[0])
        rows.append(dict(sub=s, outcome=outc, cov=len(g),
                         resid=non["abn"].mean(), focality=_gini(g["abn"].clip(lower=0))))
    F = pd.DataFrame(rows).dropna()
    z = F.copy()
    for c in ["resid", "focality"]:
        z[c] = (z[c] - z[c].mean()) / (z[c].std() + 1e-9)
    picks = {}
    for oc, name in [(1, "S"), (0, "F")]:
        grp = z[z.outcome == oc]
        c = grp[["resid", "focality"]].mean()
        d = ((grp[["resid", "focality"]] - c) ** 2).sum(axis=1)
        picks[name] = grp.loc[d.idxmin(), "sub"]
    return picks["S"], picks["F"], F.set_index("sub")


def _render_glassbrain(g, vmin, vmax, title, out_png, size_ref=None,
                       base_size=15, size_scale=70, ring_resected=True, alpha=0.9):
    """Renderiza un glass brain (4 vistas) de un set de canales a PNG."""
    from nilearn import plotting
    g = g.dropna(subset=["abn"])
    coords = g[["x", "y", "z"]].values
    r = g["rate"].values.astype(float)
    ref = size_ref if size_ref else (r.max() if r.max() > 0 else 1.0)
    sizes = base_size + size_scale * np.clip(r / ref, 0, 1)
    fig = plt.figure(figsize=(11, 3.1))
    disp = plotting.plot_markers(
        node_values=g["abn"].values, node_coords=coords, node_size=sizes,
        node_cmap="YlOrRd", node_vmin=vmin, node_vmax=vmax,
        display_mode="lyrz", figure=fig, colorbar=True, alpha=alpha)
    if ring_resected:
        res = g[g["status"].isin(RESECT)] if "status" in g else g[g["resect"] == 1]
        if len(res):
            disp.add_markers(res[["x", "y", "z"]].values, marker_color="none",
                             marker_size=150, edgecolors="#1f6fbf", linewidths=1.7, marker="o")
    fig.suptitle(title, y=1.02, fontsize=12, weight="bold")
    fig.savefig(out_png, dpi=135, bbox_inches="tight")
    plt.close(fig)


def fig_exemplars(datasets, cfg, bids_root=DEFAULT_BIDS_ROOT, fig_no=6):
    """
    Fig N: comparación S vs F de dos pacientes representativos (glass brain apilados).
    Muestra visualmente la diferencia de FORMA de la distribución de HFO.
    """
    subS, subF, feats = select_exemplars(datasets, cfg, bids_root)
    T = _abn_table(datasets, cfg, bids_root)
    both = T[T["sub"].isin([subS, subF])].dropna(subset=["abn"])
    vmin, vmax = np.nanpercentile(both["abn"], [5, 95])

    tmp = tempfile.mkdtemp()
    pS, pF = os.path.join(tmp, "s.png"), os.path.join(tmp, "f.png")

    def _desc(sub):
        f = feats.loc[sub]
        return f"residual={f['resid']:.2f}  ·  focalidad={f['focality']:.2f}  ·  {int(f['cov'])} canales"

    _render_glassbrain(T[T["sub"] == subS], vmin, vmax,
                       f"Outcome S (éxito) — {subS}   [{_desc(subS)}]", pS)
    _render_glassbrain(T[T["sub"] == subF], vmin, vmax,
                       f"Outcome F (falla) — {subF}   [{_desc(subF)}]", pF)

    fig = plt.figure(figsize=(11.5, 7.4))
    for i, p in enumerate([pS, pF]):
        ax = fig.add_axes([0.02, 0.52 - 0.48 * i, 0.96, 0.42])
        ax.imshow(mpimg.imread(p)); ax.axis("off")
    fig.text(0.5, 0.985, f"Fig {fig_no}. Distribución de HFO en el cerebro — S vs F",
             ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.045,
             "Se observa la FORMA del patrón (concentrado vs difuso, cuánto queda fuera "
             "de lo resecado), no solo el solapamiento.\n" + FSAVERAGE_CAVEAT,
             ha="center", fontsize=8.5, color="#444", wrap=True)
    return fig, (subS, subF)


def fig_group_atlas(datasets, cfg, bids_root=DEFAULT_BIDS_ROOT, fig_no=7):
    """
    Fig N: atlas grupal — electrodos de todos los pacientes S (arriba) y F (abajo),
    coloreados por anormalidad, para ver el patrón espacial del HFO por outcome.
    """
    T = _abn_table(datasets, cfg, bids_root).dropna(subset=["abn"])
    vmin, vmax = np.nanpercentile(T["abn"], [5, 95])
    tmp = tempfile.mkdtemp()
    ps = {}
    for oc, name in [(1, "S"), (0, "F")]:
        g = T[T["outcome"] == oc].copy()
        out = os.path.join(tmp, f"{name}.png")
        # marcadores chicos y uniformes, sin aros (patrón espacial de anormalidad grupal)
        _render_glassbrain(
            g, vmin, vmax,
            f"Grupo Outcome {name}  ({g['sub'].nunique()} pac., {len(g)} electrodos)",
            out, base_size=11, size_scale=0, ring_resected=False, alpha=0.6)
        ps[name] = out

    fig = plt.figure(figsize=(11.5, 7.4))
    for i, name in enumerate(["S", "F"]):
        ax = fig.add_axes([0.02, 0.52 - 0.48 * i, 0.96, 0.42])
        ax.imshow(mpimg.imread(ps[name])); ax.axis("off")
    fig.text(0.5, 0.985, f"Fig {fig_no}. Atlas grupal de anormalidad de HFO (S vs F)",
             ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.045,
             "Patrón espacial de la anormalidad de HFO agregado por grupo de outcome "
             "(color = anormalidad).\nElectrodos en espacio fsaverage sobre molde MNI152 "
             "(aprox. cualitativa).", ha="center", fontsize=8.5, color="#444", wrap=True)
    return fig
