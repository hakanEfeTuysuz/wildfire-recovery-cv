import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from het_core import load_dataset, condition_matrix, SpatialFolds, oof_predict

d = load_dataset()
x_c, y_c = d["x_coord"], d["y_coord"]
y = d["nbr_2026"].astype("float64")
X = condition_matrix(d)
elev = d["elevation"]


def spread_ratio(v, cond, nb=10):
    edges = np.quantile(cond, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, cond, side="right") - 1, 0, nb - 1)
    s = np.array([v[idx == b].std() for b in range(nb)])
    return float(s.max() / s.min())


print("=== 1) Blok boyutu duyarlılığı ===")
rows = []
for block in (1000, 2000, 3000, 5000):
    print(f"  blok {block} m ...")
    folds = SpatialFolds(x_c, y_c, block=block)
    exp = oof_predict(folds, X, y)
    res = y - exp
    sig = np.maximum(oof_predict(folds, X, np.abs(res)) * 1.2533, 0.01)
    het_s = res / sig
    rows.append({
        "blok (m)": block,
        "mekansal CV R²": round(r2_score(y, exp), 3),
        "eğitimde kalan piksel oranı": round(float(np.mean([m.mean() for m in folds.train_masks])), 2),
        "Het-σ std": round(float(het_s.std()), 3),
        "σ max/min (yükseklik)": round(spread_ratio(het_s, elev), 2),
    })
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 2) Konformal kalibrasyon (%80 aralık, CQR) ===")
folds = SpatialFolds(x_c, y_c, block=1000)
q10 = oof_predict(folds, X, y, loss="quantile", quantile=0.10)
q90 = oof_predict(folds, X, y, loss="quantile", quantile=0.90)
lo, hi = np.minimum(q10, q90), np.maximum(q10, q90)

rng = np.random.default_rng(7)
ub = np.unique(folds.block_id)
half = rng.permutation(ub)[: len(ub) // 2]
splits = {
    "rastgele blok yarısı": np.isin(folds.block_id, half),
    "batı yarısı → doğu yarısı": x_c < np.median(x_c),
}
for name, calib in splits.items():
    test = ~calib
    s = np.maximum(lo[calib] - y[calib], y[calib] - hi[calib])
    n = int(calib.sum())
    delta = float(np.quantile(s, min(1.0, 0.8 * (1 + 1 / n))))
    raw = np.mean((y[test] >= lo[test]) & (y[test] <= hi[test]))
    adj = np.mean((y[test] >= lo[test] - delta) & (y[test] <= hi[test] + delta))
    w_raw = float(np.mean(hi[test] - lo[test]))
    print(f"  {name}: düzeltme δ = {delta:.4f}")
    print(f"    test yarısında kapsama: ham %{100 * raw:.1f} → düzeltilmiş %{100 * adj:.1f} (hedef %80)")
    print(f"    ortalama aralık genişliği: {w_raw:.3f} → {w_raw + 2 * delta:.3f}")