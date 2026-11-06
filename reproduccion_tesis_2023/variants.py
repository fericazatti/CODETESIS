"""¿Qué variante de cálculo de HRR reproduce la Tabla 4 completa de la tesis? (run01, cohorte tesis24)"""
import glob, json, pickle
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from analyze import TESIS24
RS = ["resect", "resect,soz"]
th = pd.read_csv("thesis_table4.csv").set_index("C")

def hrr(df, bads, v):
    df = df.drop(df[df["colors"].isna()].index)
    if v.get("bads", True):
        df = df[~df["ch_split"].isin(bads)]
    lab = RS + (["soz"] if v.get("soz") else [])
    n = df["status"].isin(lab).sum()
    rs = {"round": round(n / 2), "ceil": int(np.ceil(n / 2)), "floor": n // 2}[v.get("half", "round")]
    if rs == 0:
        return np.nan
    key = df["counts"].astype(float) if v.get("float") else df["counts"]
    top = df.assign(k=key).sort_values("k", ascending=False, kind=v.get("kind", "quicksort")).head(rs)
    r = top["status"].isin(lab).sum() / rs
    return round(r, 2) if v.get("rnd", True) else r

data = []
for p in sorted(glob.glob("out/run01/*.pickle")):
    ds = pickle.load(open(p, "rb")); sub = ds.attrs["his_id"].replace("sub-", "")
    if sub in TESIS24:
        data.append((sub, ds.attrs["outcome"], json.loads(ds.attrs["bads"]),
                     [ds[v].to_pandas().T.reset_index() for v in ds.data_vars]))
variants = {"original": {}, "sin_bads": {"bads": False}, "ceil": {"half": "ceil"},
            "floor": {"half": "floor"}, "soz_cuenta": {"soz": True}, "sin_redondeo": {"rnd": False},
            "orden_float": {"float": True}, "orden_estable": {"float": True, "kind": "stable"},
            "sin_bads+ceil": {"bads": False, "half": "ceil"}}
for name, v in variants.items():
    eq = 0
    for C in range(1, 43):
        s = [hrr(d[3][C - 1], d[2], v) for d in data if d[1] == "S"]
        f = [hrr(d[3][C - 1], d[2], v) for d in data if d[1] == "F"]
        s = [x for x in s if x == x]; f = [x for x in f if x == x]
        r = mannwhitneyu(s, f, alternative="two-sided", method="asymptotic", use_continuity=False)
        eq += min(r.statistic, len(s) * len(f) - r.statistic) == th.loc[C, "U_tesis"]
    print(f"{name:16s} U idéntico a la tesis en {eq}/42")
