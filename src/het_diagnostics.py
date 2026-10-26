import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics import r2_score
from het_core import load_dataset, condition_matrix, SpatialFolds, oof_predict, percentile_from_quantiles

d = load_dataset()
x_c, y_c = d["x_coord"], d["y_coord"]
y = d["nbr_2026"].astype("float64")
X = condition_matrix(d)
pre, elev, dnbr = d["baseline_vigor"], d["elevation"], d["dnbr_value"]
print(f"{len(y):,} piksel yüklendi")

folds = SpatialFolds(x_c, y_c)

print("1/4 beklenen NBR (ortalama model)...")
expected = oof_predict(folds, X, y)
resid = y - expected
r2 = r2_score(y, expected)
het_z = resid / resid.std()
print(f"    mekansal CV R² = {r2:.3f}")

print("2/4 koşullu ölçek (|artık| modeli)...")
sigma = np.maximum(oof_predict(folds, X, np.abs(resid)) * 1.2533, 0.01)   # E|e| = σ·√(2/π)
het_s = resid / sigma

print("3/4 koşullu kantiller (7 model x 5 kat, birkaç dakika sürer)...")
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
cols = []
for q in QS:
    print(f"    kantil {q}")
    cols.append(oof_predict(folds, X, y, loss="quantile", quantile=q))
qpred = np.sort(np.column_stack(cols), axis=1)
het_p = percentile_from_quantiles(y, qpred, QS)

print("\n=== Kantil kalibrasyonu (mekansal CV; ideal: ikisi eşit) ===")
for j, q in enumerate(QS):
    print(f"  hedef %{100*q:4.0f}  gerçekleşen %{100*np.mean(y < qpred[:, j]):5.1f}")
cov = np.mean((y >= qpred[:, 1]) & (y <= qpred[:, 5]))
print(f"  %10-%90 aralığı kapsama: %{100*cov:.1f} (hedef %80)")


def bin_stds(v, cond, nb=10):
    edges = np.quantile(cond, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, cond, side="right") - 1, 0, nb - 1)
    return np.array([v[idx == b].std() for b in range(nb)])


variants = [("Het-z (genel σ)", het_z), ("Het-σ (koşullu σ)", het_s), ("Het-P (koşullu persentil)", het_p)]
rows = []
for name, v in variants:
    row = {"varyant": name}
    for cname, cond in [("yangın öncesi NBR", pre), ("yükseklik", elev), ("dNBR", dnbr)]:
        s = bin_stds(v, cond)
        row[f"σ max/min ({cname})"] = round(float(s.max() / s.min()), 2)
    rows.append(row)
het_tab = pd.DataFrame(rows)
print("\n=== Koşul gruplarına göre yayılım eşitliği (1,00 ideal) ===")
print(het_tab.to_string(index=False))
het_tab.to_csv("outputs/het_diagnostics.csv", index=False)

print("\n4/4 mekansal yapı (varyogram) ve yamalar...")
CELL = 10
ix = np.rint((x_c - x_c.min()) / CELL).astype(np.int64)
iy = np.rint((y_c.max() - y_c) / CELL).astype(np.int64)
H, W = int(iy.max()) + 1, int(ix.max()) + 1


def to_grid(v):
    g = np.full((H, W), np.nan, dtype=np.float32)
    g[iy, ix] = v
    return g


LAGS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80]   # hücre = 10 m ... 800 m


def semivar(g, h):
    out = []
    for a, b in ((g[:, :-h], g[:, h:]), (g[:-h, :], g[h:, :])):
        dd = a - b
        out.append(0.5 * float(np.mean(dd[np.isfinite(dd)] ** 2)))
    return float(np.mean(out))


vario = {}
for name, v in variants:
    g = to_grid(v.astype("float32"))
    var = float(np.nanvar(g))
    vario[name] = np.array([semivar(g, h) / var for h in LAGS])
    del g

lag_m = [h * CELL for h in LAGS]
print("\nNormalleştirilmiş yarı-varyans (gamma / varyans):")
print(pd.DataFrame(vario, index=[f"{m} m" for m in lag_m]).round(3).to_string())

for name in vario:
    gm = vario[name]
    plateau = 1.0
    nugget = float(gm[0] / plateau)
    rng_m = next((m for m, val in zip(lag_m, gm) if val >= 0.95 * plateau), lag_m[-1])
    print(f"\n{name}: 10 m altı pay (gürültü + alt-piksel) ≈ %{100*nugget:.0f}, "
          f"mekansal yapılı pay ≈ %{100*(1-nugget):.0f}, yaklaşık menzil ≈ {rng_m} m")

nug = float(vario["Het-z (genel σ)"][0])
print("\n=== Varyans bütçesi (2026 NBR'sinin toplam varyansı) ===")
print(f"  koşullarla açıklanan            %{100*r2:.0f}")
print(f"  mekansal yapılı sapma (Het)     %{100*(1-r2)*(1-nug):.0f}")
print(f"  10 m altı değişkenlik/gürültü   %{100*(1-r2)*nug:.0f}")

plt.figure(figsize=(8, 5))
for name, gm in vario.items():
    plt.plot(lag_m, gm, marker="o", label=name)
plt.axhline(1, color="gray", linestyle=":")
plt.xscale("log")
plt.xlabel("Uzaklık (m)")
plt.ylabel("Yarı-varyans / varyans")
plt.title("Het varyantlarının mekansal yapısı")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/het_variogram.png", dpi=150)


def patch_stats(v, thr=-1.0):
    g = to_grid(v.astype("float32"))
    valid = np.isfinite(g)
    num = ndimage.uniform_filter(np.where(valid, g, 0).astype("float32"), size=5, mode="constant")
    den = ndimage.uniform_filter(valid.astype("float32"), size=5, mode="constant")
    sm = np.where(den > 0.6, num / np.maximum(den, 1e-6), np.nan)
    lab, n = ndimage.label(np.isfinite(sm) & (sm < thr))
    ha = np.bincount(lab.ravel())[1:] * (CELL * CELL) / 10000.0 if n else np.array([])
    return ha, float(valid.sum() * CELL * CELL / 10000.0)


print("\n=== Geride kalan yamalar (Het-σ, 50 m ortalama < -1) ===")
rng = np.random.default_rng(1)
for label, v in [("gerçek", het_s), ("karıştırılmış (gürültü referansı)", rng.permutation(het_s))]:
    ha, total = patch_stats(v)
    big = ha[ha >= 1]
    print(f"  {label}: >=1 ha yama sayısı={len(big)}, >=5 ha={int((ha >= 5).sum())}, "
          f"toplam alan={big.sum():.0f} ha (geçerli alanın %{100*big.sum()/total:.2f}'i), "
          f"en büyük={ha.max() if len(ha) else 0:.1f} ha")
print("\nKaydedildi: outputs/het_diagnostics.csv, outputs/het_variogram.png")