"""Tabla 4 de la tesis (Mann-Whitney HRR S vs F, 42 configs) en distintos escenarios.

HRR calculado EXACTAMENTE como analysis.py @ main (2023):
  - se descartan canales cuyo ánodo es 'bad',
  - zona HFO = top round(N_resect/2) canales por 'counts' (orden de pandas sobre dtype object),
  - HRR = fracción de esa zona dentro de la resección, redondeado a 2 decimales.
Mann-Whitney bilateral, aproximación normal con corrección por empates y sin corrección de
continuidad (equivalente a la "Sig. asintótica" de SPSS).

Escenarios: modo {run01 (protocolo original), allruns (todos los runs)} x cohorte
{tesis24 (Tabla 1), tesis24+6 excluidos en 2023, todos los SEEG}.
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TESIS24 = ["HUP117", "HUP133", "HUP134", "HUP138", "HUP144", "HUP146", "HUP148", "HUP157",
           "HUP160", "HUP162", "HUP163", "HUP164", "HUP165", "HUP166", "HUP171", "HUP172",
           "HUP177", "HUP179", "HUP180", "HUP181", "HUP185", "HUP187", "HUP188", "HUP190"]
EXCL6 = ["HUP130", "HUP135", "HUP139", "HUP140", "HUP141", "HUP142"]
RESECT = ["resect", "resect,soz"]


def hrr_original(df, bads):
    df = df.drop(df[df["colors"].isna()].index)
    for bad in bads:
        df = df.drop(df[df["ch_split"] == bad].index)
    rs = round(df["status"].isin(RESECT).sum() / 2)
    if rs == 0:
        return np.nan
    top = df.sort_values("counts", ascending=False).head(rs)
    return round(top["status"].isin(RESECT).sum() / rs, 2)


def load_hrr(pickles, bads_from=None):
    rows = []
    for p in pickles:
        ds = pickle.load(open(p, "rb"))
        sub = str(ds.attrs["his_id"]).replace("sub-", "").strip()
        bads = json.loads(ds.attrs["bads"]) if "bads" in ds.attrs else bads_from(sub)
        for i, v in enumerate(ds.data_vars):
            rows.append(dict(subject=sub, outcome=str(ds.attrs["outcome"]).strip(), C=i + 1,
                             cfg=v, hrr=hrr_original(ds[v].to_pandas().T.reset_index(), bads)))
    return pd.DataFrame(rows)


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def mw_table(R, subjects=None):
    if subjects is not None:
        R = R[R.subject.isin(subjects)]
    out = []
    for (c, cfg), g in R.dropna(subset=["hrr"]).groupby(["C", "cfg"]):
        s = g.loc[g.outcome == "S", "hrr"]
        f = g.loc[g.outcome == "F", "hrr"]
        if len(s) < 2 or len(f) < 2:
            continue
        r = mannwhitneyu(s, f, alternative="two-sided", method="asymptotic", use_continuity=False)
        out.append(dict(C=c, cfg=cfg, nS=len(s), nF=len(f),
                        U=min(r.statistic, len(s) * len(f) - r.statistic),
                        AUC_S_mayor=r.statistic / (len(s) * len(f)), p=r.pvalue))
    if not out:  # escenario sin datos suficientes (p.ej. corrida aún parcial)
        return pd.DataFrame(columns=["C", "cfg", "nS", "nF", "U", "AUC_S_mayor", "p",
                                     "p_bonf", "q_fdr"])
    T = pd.DataFrame(out).sort_values("C")
    T["p_bonf"] = np.minimum(T.p * len(T), 1)
    T["q_fdr"] = bh(T.p)
    return T


def main():
    os.makedirs(RES, exist_ok=True)
    tables, summary = [], []

    # Referencia: datos archivados de 2023 (23 pickles de git + HUP117 del cuaderno, C33-C42)
    chtsv = lambda s: list(pd.read_csv(  # noqa: E731
        glob.glob(os.path.join(HERE, "bids", "ds004100", f"sub-{s}", "ses-presurgery", "ieeg",
                               f"sub-{s}_*task-interictal*run-01_channels.tsv"))[0],
        sep="\t").query("status == 'bad'")["name"])
    arch = load_hrr(sorted(glob.glob(os.path.join(HERE, "pickles_2023", "tesis", "*.pickle"))),
                    bads_from=chtsv)
    nb = pd.read_csv(os.path.join(HERE, "nb_hup117.csv"))
    arch = pd.concat([arch[arch.subject != "HUP150"],
                      nb.assign(subject="HUP117", outcome="S",
                                cfg=nb.C.map(dict(zip(arch.C, arch.cfg))))])
    T = mw_table(arch[arch.C >= 33], TESIS24).assign(modo="2023_archivado", cohorte="tesis24")
    # solo hay C33-C42 (HUP117 sobrevive en el cuaderno solo para esas): la familia real son
    # 42 tests -> Bonferroni x42; FDR no es calculable sin los 42 p-valores.
    T["p_bonf"] = np.minimum(T.p * 42, 1)
    T["q_fdr"] = np.nan
    tables.append(T)

    for mode in ("run01", "allruns"):
        pk = sorted(glob.glob(os.path.join(HERE, "out", mode, "*.pickle")))
        if not pk:
            continue
        R = load_hrr(pk)
        R.to_csv(os.path.join(RES, f"hrr_{mode}.csv"), index=False)
        for name, subs in (("tesis24", TESIS24), ("tesis24+excl6", TESIS24 + EXCL6),
                           ("todos", None)):
            tables.append(mw_table(R, subs).assign(modo=mode, cohorte=name))
        if mode == "run01":  # control: ¿el run01 de hoy reproduce los HRR de 2023?
            m = R.merge(arch, on=["subject", "C"], suffixes=("_hoy", "_2023"))
            m = m[m.subject != "HUP117"]
            summary.append(f"Control run01 vs 2023 (23 sujetos x 42 configs): "
                           f"{(m.hrr_hoy == m.hrr_2023).sum()}/{len(m)} HRR idénticos")

    T = pd.concat(tables, ignore_index=True)
    T.to_csv(os.path.join(RES, "mann_whitney_escenarios.csv"), index=False)
    th = pd.read_csv(os.path.join(HERE, "thesis_table4.csv"))
    key = T[T.C.isin([36, 37, 40])].merge(th, on="C")
    piv = key.assign(val=key.apply(lambda r: f"p={r.p:.3f} ({r.nS}S/{r.nF}F)", axis=1)) \
             .pivot_table(index=["modo", "cohorte"], columns="C", values="val", aggfunc="first")
    Tv = T.dropna(subset=["p"])
    best = Tv.loc[Tv.groupby(["modo", "cohorte"]).p.idxmin(),
                 ["modo", "cohorte", "C", "cfg", "nS", "nF", "U", "p", "p_bonf", "q_fdr"]]
    with open(os.path.join(RES, "RESUMEN.md"), "w") as f:
        f.write("# Configuraciones de la tesis (C36, C37, C40) por escenario\n\n")
        f.write("Tesis (SPSS): C36 p=0.05, C37 p=0.04, C40 p=0.06\n\n")
        f.write(piv.to_markdown() + "\n\n# Mejor configuración por escenario (de 42)\n\n")
        f.write(best.round(4).to_markdown(index=False) + "\n\n")
        f.write("\n".join(summary) + "\n")
    print(open(os.path.join(RES, "RESUMEN.md")).read())


if __name__ == "__main__":
    main()
