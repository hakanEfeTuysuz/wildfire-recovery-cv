import numpy as np
import pandas as pd
from itertools import combinations
from scipy import ndimage
from het_core import load_dataset, condition_matrix, SpatialFolds, oof_predict

CELL = 10
DEEP_ITER = 200
# Süzgeç pencereleri (hücre): 50 m, 1 km, 3 km
W_FINE, W_MID, W_WIDE = 5, 101, 301
DATES = ["2026-08-20", "2026-07-24", "2026-08-03"]

d = load_dataset()
x_c, y_c = d["x_coord"], d["y_coord"]
X = condition_matrix(d)

series = {"2026-08-20": d["nbr_2026"].astype("float64")}
for dt in DATES[1:]:
    series[dt] = np.load(f"outputs/nbr_cache_{dt}.npy").astype("float64")

valid = np.ones(len(x_c), dtype=bool)
for v in series.values():
    valid &= np.isfinite(v)
print(f"Tüm tarihlerde geçerli piksel: {int(valid.sum()):,} / {len(valid):,}")

xv, yv, Xv = x_c[valid], y_c[valid], X[valid]
folds = SpatialFolds(xv, yv)

xmin, ymax = x_c.min(), y_c.max()
ix = np.rint((xv - xmin) / CELL).astype(np.int64)
iy = np.rint((ymax - yv) / CELL).astype(np.int64)
H = int(np.rint((ymax - y_c.min()) / CELL)) + 1
W = int(np.rint((x_c.max() - xmin) / CELL)) + 1


def smooth(g, size):
    ok = np.isfinite(g)
    num = ndimage.uniform_filter(np.where(ok, g, 0).astype("float32"), size=size, mode="constant")
    den = ndimage.uniform_filter(ok.astype("float32"), size=size, mode="constant")
    return np.where(den > 0.2, num / np.maximum(den, 1e-6), np.nan).astype("float32")


def components(resid):
    g = np.full((H, W), np.nan, dtype="float32")
    g[iy, ix] = resid
    s_fine = smooth(g, W_FINE)
    s_mid = smooth(g, W_MID)
    s_wide = smooth(g, W_WIDE)
    comps = {
        "< 50 m (piksel gürültüsü + alt-piksel)": (g - s_fine)[iy, ix],
        "50 m - 1 km (yerel)": (s_fine - s_mid)[iy, ix],
        "1 - 3 km (çevresel)": (s_mid - s_wide)[iy, ix],
        "> 3 km (bölgesel)": s_wide[iy, ix],
    }
    del g, s_fine, s_mid, s_wide
    return comps


per_date = {}
for dt, v in series.items():
    print(f"beklenen NBR ve ölçek ayrıştırması: {dt}")
    vs = v[valid]
    resid = vs - oof_predict(folds, Xv, vs, max_iter=DEEP_ITER)
    per_date[dt] = (resid, components(resid))

names = list(per_date[DATES[0]][1].keys())
ok = np.ones(len(xv), dtype=bool)
for dt in per_date:
    for c in per_date[dt][1].values():
        ok &= np.isfinite(c)
print(f"Tüm bileşenlerde geçerli piksel: {int(ok.sum()):,}")

resid0, comps0 = per_date[DATES[0]]
var_total = float(np.var(resid0[ok]))
rows = []
for n in names:
    rs = [np.corrcoef(per_date[a][1][n][ok], per_date[b][1][n][ok])[0, 1] for a, b in combinations(DATES, 2)]
    rows.append({
        "ölçek bileşeni": n,
        "artık varyansındaki pay": round(float(np.var(comps0[n][ok])) / var_total, 3),
        "tarihler arası r": round(float(np.mean(rs)), 3),
    })
tab = pd.DataFrame(rows)
print("\n=== Het'in ölçek bileşenleri: pay ve tekrarlanabilirlik ===")
print(tab.to_string(index=False))
tab.to_csv("outputs/het_scales.csv", index=False)
print("\n(Payların toplamı 1'e yakın olmalı; bileşenler birbiriyle az da olsa ilişkili olabilir.)")