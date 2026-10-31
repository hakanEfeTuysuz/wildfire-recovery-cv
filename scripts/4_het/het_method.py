"""Het göstergesi: koşullara göre beklenen NBR'den standartlaştırılmış sapma.
Bilinen toparlanma göstergeleriyle koşullara bağımlılık açısından karşılaştırma ve Het haritası."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from wildfire.config import FIGURES_DIR, TABLES_DIR
from wildfire.het import load_dataset, SpatialFolds, oof_predict
from wildfire.severity import SEVERITY_LABELS, BURNED_CLASSES

INCLUDE_SEVERITY = True   # Het beklentisi yanık şiddetine (dNBR) de koşullansın mı

data = load_dataset()
pre = data["baseline_vigor"].astype("float64")
dnbr = data["dnbr_value"].astype("float64")
y = data["nbr_2026"].astype("float64")
elev = data["elevation"]
slope = data["slope_deg"]
ns = data["ns_gradient"]
x_c = data["x_coord"]
y_c = data["y_coord"]
sev = data["severity_class"]
post = pre - dnbr         # yangın hemen sonrası NBR (dNBR = öncesi - sonrası)

# --- Koşul değişkenleri ---
cond_all = {
    "yangın öncesi NBR": pre,
    "yükseklik": elev,
    "eğim": slope,
    "kuzey-güney eğim": ns,
    "dNBR (şiddet)": dnbr,
}
X_all = np.column_stack(list(cond_all.values())).astype("float32")
if INCLUDE_SEVERITY:
    X_het = X_all
else:
    X_het = np.column_stack([pre, elev, slope, ns]).astype("float32")

# --- Tamponlu mekansal katlar (test bloğu + 8 komşusu eğitimden çıkarılır) ---
folds = SpatialFolds(x_c, y_c)

# --- Het göstergesi ---
print("Het için beklenen NBR hesaplanıyor (mekansal çapraz doğrulama)...")
expected = oof_predict(folds, X_het, y)
print(f"Beklenen NBR modelinin mekansal CV R²'si: {r2_score(y, expected):.3f}")
resid = y - expected
het = resid / np.std(resid)

# --- Karşılaştırılacak bilinen göstergeler ---
ham = y - post                                  # ham NBR artışı
oran = ham / dnbr                               # göreli toparlanma oranı (kaybın ne kadarı geri geldi)
lo, hi = np.percentile(oran, [1, 99])
oran = np.clip(oran, lo, hi)                    # uç değerleri kırp

indicators = {
    "Het (koşullu sapma)": het,
    "Ham toparlanma (NBR artışı)": ham,
    "Göreli toparlanma oranı": oran,
    "2026 NBR (ham düzey)": y,
}

rows = []
for name, ind in indicators.items():
    print(f"Karşılaştırma: {name}")
    ind = np.asarray(ind, dtype="float64")
    r_vals = {c: float(np.corrcoef(v, ind)[0, 1]) for c, v in cond_all.items()}
    pred = oof_predict(folds, X_all, ind)
    row = {"gösterge": name}
    row.update({f"r({c})": round(r, 3) for c, r in r_vals.items()})
    row["max |r|"] = round(max(abs(r) for r in r_vals.values()), 3)
    row["koşullarla açıklanan varyans (mekansal CV R²)"] = round(r2_score(ind, pred), 3)
    rows.append(row)

table = pd.DataFrame(rows)
out_csv = TABLES_DIR / "het_comparison.csv"
table.to_csv(out_csv, index=False)
print("\n=== Göstergelerin koşullara bağımlılığı ===")
print(table.to_string(index=False))

# --- Şiddet sınıfına göre Het özeti ---
labels = {c: SEVERITY_LABELS[c] for c in BURNED_CLASSES}
print("\n=== Het özeti (şiddet sınıfına göre) ===")
for c, lab in labels.items():
    m = sev == c
    if m.sum() == 0:
        continue
    print(f"{lab:20s} ortalama Het={het[m].mean():+.3f}   geride kalan (Het<-1) oranı=%{100 * np.mean(het[m] < -1):.1f}")

# --- Het haritası (100 m hücrelerde ortalama) ---
CELL = 100
ix = ((x_c - x_c.min()) // CELL).astype(np.int64)
iy = ((y_c.max() - y_c) // CELL).astype(np.int64)
W, H = ix.max() + 1, iy.max() + 1
flat = iy * W + ix
s = np.bincount(flat, weights=het, minlength=H * W)
n = np.bincount(flat, minlength=H * W)
grid = np.where(n > 0, s / np.maximum(n, 1), np.nan).reshape(H, W)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(grid, cmap="RdBu", vmin=-2, vmax=2)
ax.set_title("Het göstergesi: benzer koşullardaki alanlara göre toparlanma sapması")
ax.axis("off")
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Het (z): + beklenenden iyi, − geride")
plt.tight_layout()
out_png = FIGURES_DIR / "het_map.png"
plt.savefig(out_png, dpi=150)
print(f"\nKaydedildi: {out_csv} ve {out_png}")
