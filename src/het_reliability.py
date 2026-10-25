import os
import numpy as np
import pandas as pd
import xarray as xr
from itertools import combinations
from common import get_sentinel2_items, compute_nbr_and_mask
from het_core import load_dataset, condition_matrix, SpatialFolds, oof_predict

DATES = ["2026-07-24", "2026-08-03"]   # npz'deki 2026-08-20 ile birlikte 3 tarih
d = load_dataset()
x_c, y_c = d["x_coord"], d["y_coord"]
X = condition_matrix(d)
dnbr = d["dnbr_value"].astype("float64")
post = d["baseline_vigor"].astype("float64") - dnbr

series = {"2026-08-20": d["nbr_2026"].astype("float64")}
xs, ys = xr.DataArray(x_c, dims="p"), xr.DataArray(y_c, dims="p")

for date in DATES:
    cache = f"outputs/nbr_cache_{date}.npy"
    if os.path.exists(cache):
        series[date] = np.load(cache).astype("float64")
        print(f"{date}: önbellekten okundu")
        continue
    print(f"--- {date} indiriliyor ---")
    nbr, good = compute_nbr_and_mask(get_sentinel2_items(date))
    sel = nbr.sel(x=xs, y=ys, method="nearest")
    off = np.maximum(np.abs(sel.x.values - x_c), np.abs(sel.y.values - y_c))
    print(f"    koordinat sapması: medyan {np.median(off):.1f} m, en büyük {off.max():.1f} m")
    ok = good.sel(x=xs, y=ys, method="nearest").values.astype(bool)
    v = np.where(ok & np.isfinite(sel.values), sel.values, np.nan).astype("float32")
    np.save(cache, v)
    series[date] = v.astype("float64")

valid = np.ones(len(x_c), dtype=bool)
for v in series.values():
    valid &= np.isfinite(v)
print(f"Tüm tarihlerde geçerli piksel: {int(valid.sum()):,} / {len(valid):,}")

folds = SpatialFolds(x_c[valid], y_c[valid])
Xs, dn, po = X[valid], dnbr[valid], post[valid]
het, ham, level = {}, {}, {}
for dt, v in series.items():
    print(f"beklenen NBR modeli: {dt}")
    vs = v[valid]
    r = vs - oof_predict(folds, Xs, vs)
    het[dt] = r / r.std()
    ham[dt] = vs - po
    level[dt] = vs

ix = np.rint((x_c[valid] - x_c.min()) / 10).astype(np.int64)
iy = np.rint((y_c.max() - y_c[valid]) / 10).astype(np.int64)
Wc = int(ix.max()) + 1


def block_means(v, s):
    cid = (iy // s) * (Wc // s + 1) + (ix // s)
    cnt = np.bincount(cid)
    tot = np.bincount(cid, weights=v)
    ok = cnt >= max(1, int(0.5 * s * s))
    return tot[ok] / cnt[ok]


pairs = list(combinations(series.keys(), 2))
rows = []
for name, store in [("Het", het), ("Ham toparlanma", ham), ("2026 NBR düzeyi", level)]:
    for s in (1, 5, 10, 25):   # 10 m, 50 m, 100 m, 250 m
        rs = [np.corrcoef(block_means(store[a], s), block_means(store[b], s))[0, 1] for a, b in pairs]
        rows.append({"gösterge": name, "ölçek (m)": s * 10, "tarihler arası ortalama r": round(float(np.mean(rs)), 3)})
tab = pd.DataFrame(rows)
print("\n=== Test-tekrar test güvenilirliği (aynı yaz, farklı tarihler) ===")
print(tab.to_string(index=False))
tab.to_csv("outputs/het_reliability.csv", index=False)
print("\nKaydedildi: outputs/het_reliability.csv")