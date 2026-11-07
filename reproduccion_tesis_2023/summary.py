"""Resumen compacto de results/mann_whitney_escenarios.csv (C36/C37/C40 y mejor config)."""
import sys
import pandas as pd
T = pd.read_csv("results/mann_whitney_escenarios.csv")
modes = sys.argv[1:] or list(T.modo.unique())
for (m, c), g in T[T.modo.isin(modes)].groupby(["modo", "cohorte"], sort=False):
    k = g.set_index("C")
    b = g.loc[g.p.idxmin()]
    cell = lambda C: f"p={k.p[C]:.3f}" if C in k.index else "n/a"  # noqa: E731
    q = "" if pd.isna(b.q_fdr) else f" q_FDR={b.q_fdr:.2f}"
    print(f"{m:15s} {c:14s} n={int(g.nS.max())}S/{int(g.nF.max())}F | C36 {cell(36)} | "
          f"C37 {cell(37)} | C40 {cell(40)} | mejor C{int(b.C)} p={b.p:.3f} "
          f"Bonf={b.p_bonf:.2f}{q} | p<0.05 en {(g.p < 0.05).sum()}/{len(g)}")
