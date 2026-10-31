"""Model türü x blok boyutu mekansal beceri eğrisi ve Het-σ'nın yığılmış / iç içe
(sızıntısız) hesabının karşılaştırması."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wildfire.config import TABLES_DIR
from wildfire.het import (
    load_dataset, condition_matrix, SpatialFolds, oof_predict, oof_with_baseline, spread_ratio, MAX_TRAIN, SEED,
)

BLOCK_SIZES = (1000, 2000, 3000, 5000)
DEEP_ITER = 200
SHALLOW_ITER = 100

d = load_dataset()
x_c, y_c = d["x_coord"], d["y_coord"]
y = d["nbr_2026"].astype("float64")
X = condition_matrix(d)
elev, pre = d["elevation"], d["baseline_vigor"]


def deep():
    return HistGradientBoostingRegressor(max_iter=DEEP_ITER, learning_rate=0.1, max_depth=6, random_state=SEED)


def shallow():
    return HistGradientBoostingRegressor(max_iter=SHALLOW_ITER, learning_rate=0.1, max_depth=3, random_state=SEED)


def linear():
    return make_pipeline(StandardScaler(), LinearRegression())


MODELS = {"Doğrusal": linear, "Sığ ağaç (derinlik 3)": shallow, "Derin ağaç (Het'te kullanılan)": deep}


print("=== 1) Mekansal beceri eğrisi: model türü x blok boyutu ===")
sst = float(np.sum((y - y.mean()) ** 2))
rows = []
for block in BLOCK_SIZES:
    folds = SpatialFolds(x_c, y_c, block=block)
    for name, mk in MODELS.items():
        print(f"  blok {block} m, {name} ...")
        pred, base = oof_with_baseline(folds, X, y, mk)
        sse = float(np.sum((y - pred) ** 2))
        sse_base = float(np.sum((y - base) ** 2))
        rows.append({
            "blok (m)": block,
            "model": name,
            "R² (genel ortalamaya göre)": round(1 - sse / sst, 3),
            "beceri skoru (eğitim ortalamasına göre)": round(1 - sse / sse_base, 3),
        })
skill = pd.DataFrame(rows)
print(skill.to_string(index=False))
skill.to_csv(TABLES_DIR / "het_skill_curve.csv", index=False)


def nested_expected_sigma(folds, block=1000):
    """Her dış kat için σ modeli, o katın verisinden hiç etkilenmemiş (iç OOF) artıklarla eğitilir."""
    rng = np.random.default_rng(SEED)
    expected = np.full(len(y), np.nan)
    sigma = np.full(len(y), np.nan)
    for i in range(folds.k):
        print(f"    dış kat {i + 1}/{folds.k}")
        tr = np.flatnonzero(folds.train_masks[i])
        if len(tr) > MAX_TRAIN:
            tr = rng.choice(tr, MAX_TRAIN, replace=False)
        test = folds.fold == i
        expected[test] = deep().fit(X[tr], y[tr]).predict(X[test])
        inner = SpatialFolds(x_c[tr], y_c[tr], block=block, seed=SEED + i + 1)
        inner_pred = np.full(len(tr), np.nan)
        for j in range(inner.k):
            jj = np.flatnonzero(inner.train_masks[j])
            if len(jj) > MAX_TRAIN:
                jj = rng.choice(jj, MAX_TRAIN, replace=False)
            tj = inner.fold == j
            inner_pred[tj] = deep().fit(X[tr][jj], y[tr][jj]).predict(X[tr][tj])
        inner_abs = np.abs(y[tr] - inner_pred)
        ok = np.isfinite(inner_abs)
        s_model = deep().fit(X[tr][ok], inner_abs[ok])
        sigma[test] = np.maximum(s_model.predict(X[test]) * 1.2533, 0.01)
    return expected, sigma


print("\n=== 2) Het-σ: yığılmış ve iç içe (sızıntısız) hesabın karşılaştırması ===")
folds = SpatialFolds(x_c, y_c, block=1000)
print("  yığılmış sürüm ...")
exp_s = oof_predict(folds, X, y, max_iter=DEEP_ITER)
res_s = y - exp_s
sig_s = np.maximum(oof_predict(folds, X, np.abs(res_s), max_iter=DEEP_ITER) * 1.2533, 0.01)
het_stack = res_s / sig_s
print("  iç içe sürüm ...")
exp_n, sig_n = nested_expected_sigma(folds)
het_nest = (y - exp_n) / sig_n

cmp_rows = []
for name, v in [("Het-σ yığılmış", het_stack), ("Het-σ iç içe", het_nest)]:
    cmp_rows.append({
        "sürüm": name,
        "std": round(float(v.std()), 3),
        "σ max/min (yükseklik)": round(spread_ratio(v, elev), 2),
        "σ max/min (yangın öncesi NBR)": round(spread_ratio(v, pre), 2),
    })
print(pd.DataFrame(cmp_rows).to_string(index=False))
print(f"\nİki sürümün korelasyonu: r = {np.corrcoef(het_stack, het_nest)[0, 1]:.4f}")
print(f"Ortalama mutlak fark: {np.mean(np.abs(het_stack - het_nest)):.3f} (Het birimi)")