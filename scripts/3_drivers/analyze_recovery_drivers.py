"""Toparlanma sürücüleri (yangın öncesi NBR, şiddet, arazi) ve model/Het betiklerinin
kullandığı piksel veri setinin (data/interim/recovery_dataset.npz) üretimi."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from wildfire.config import FIGURES_DIR, RECOVERY_DATASET, TARGET_YEAR
from wildfire.imagery import (
    compute_nbr_and_mask, find_best_august_date, get_terrain_features, align_to_reference, load_fire_reference,
)
from wildfire.severity import SEVERITY_LABELS, BURNED_CLASSES

ref = load_fire_reference()
nbr_pre, nbr_post, forest_mask = ref.nbr_pre, ref.nbr_post, ref.forest_mask
valid_2021, dnbr_2021 = ref.valid, ref.dnbr

severity_labels = SEVERITY_LABELS
severity_class = ref.severity_class
burned_indices = BURNED_CLASSES

print(f"--- En güncel yıl ({TARGET_YEAR} Ağustos) ---")
best_date, items = find_best_august_date(TARGET_YEAR)
if not items:
    raise SystemExit(f"{TARGET_YEAR} Ağustos için uygun (bulutsuz) görüntü bulunamadı")
print(f"  Seçilen tarih: {best_date}")
nbr_2026, mask_2026 = compute_nbr_and_mask(items)
nbr_2026, mask_2026 = align_to_reference(nbr_2026, mask_2026, nbr_pre)
valid_2026 = (mask_2026 & forest_mask).values

recovery_amount = (nbr_2026 - nbr_post).values
baseline_vigor = nbr_pre.values

overall_valid = (
    valid_2021.values & valid_2026 & np.isin(severity_class, burned_indices)
    & np.isfinite(recovery_amount) & np.isfinite(baseline_vigor)
)

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
results = []
for ax, idx in zip(axes.flat, burned_indices):
    class_mask = overall_valid & (severity_class == idx)
    x = baseline_vigor[class_mask]
    y = recovery_amount[class_mask]
    if len(x) < 30:
        ax.set_title(f"{severity_labels[idx]} (yetersiz piksel)")
        continue
    r, p = pearsonr(x, y)
    results.append((severity_labels[idx], r, p, len(x)))
    ax.hexbin(x, y, gridsize=40, cmap="viridis", mincnt=1)
    ax.set_title(f"{severity_labels[idx]} (r={r:.2f})")
    ax.set_xlabel("Yangın öncesi NBR")
    ax.set_ylabel("Toparlanma miktarı (2026 − sonrası)")

plt.tight_layout()
out_png = FIGURES_DIR / "recovery_vs_baseline_vigor.png"
plt.savefig(out_png, dpi=150)

print("\nSınıf içi korelasyon (yangın öncesi NBR vs. toparlanma miktarı):")
for label, r, p, n in results:
    print(f"  {label:25s}: r={r:.3f}  p={p:.4f}  n={n}")
print(f"\nKaydedildi: {out_png}")

print("\nYangın öncesi NBR vs. 2026 NBR (mutlak seviye) korelasyonu:")
for idx in burned_indices:
    class_mask = overall_valid & (severity_class == idx)
    x = baseline_vigor[class_mask]
    y = nbr_2026.values[class_mask]
    if len(x) < 30:
        continue
    r2, p2 = pearsonr(x, y)
    print(f"  {severity_labels[idx]:25s}: r={r2:.3f}  p={p2:.4f}")

elevation, slope_deg, ns_gradient = get_terrain_features(target_grid=nbr_pre)

print(f"\nYükseklik aralığı (temizlik öncesi): {np.nanmin(elevation):.1f} - {np.nanmax(elevation):.1f} m")
print(f"Eğim aralığı (temizlik öncesi): {np.nanmin(slope_deg):.1f} - {np.nanmax(slope_deg):.1f} derece")

terrain_valid = (
    np.isfinite(elevation) & (elevation > -50) & (elevation < 3000)
    & np.isfinite(slope_deg) & (slope_deg < 80)
    & np.isfinite(ns_gradient)
)
final_valid = overall_valid & terrain_valid
print(f"Terrain temizliği sonrası geçerli piksel: {int(final_valid.sum())} / {int(overall_valid.sum())}")

# Mekansal blok ayrımı için piksel koordinatları (UTM, metre cinsinden)
xx, yy = np.meshgrid(nbr_pre.x.values, nbr_pre.y.values)

np.savez(
    RECOVERY_DATASET,
    baseline_vigor=baseline_vigor[final_valid],
    dnbr_value=dnbr_2021.values[final_valid],
    severity_class=severity_class[final_valid],
    nbr_2026=nbr_2026.values[final_valid],
    elevation=elevation[final_valid],
    slope_deg=slope_deg[final_valid],
    ns_gradient=ns_gradient[final_valid],
    x_coord=xx[final_valid],
    y_coord=yy[final_valid],
    nbr_2026_date=np.array(str(best_date)),
)
print(f"Kaydedildi: {RECOVERY_DATASET} (model için, koordinatlar dahil)")
