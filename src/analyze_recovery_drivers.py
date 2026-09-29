import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from common import get_sentinel2_items, compute_nbr_and_mask, get_forest_mask, find_best_august_date, get_terrain_features

print("--- Yangın öncesi (baseline / arazi kalitesi vekili) ---")
pre_items = get_sentinel2_items("2021-07-20")
nbr_pre, mask_pre = compute_nbr_and_mask(pre_items)
forest_mask = get_forest_mask(target_grid=nbr_pre)

print("--- Yangın hemen sonrası ---")
post_items = get_sentinel2_items("2021-08-14")
nbr_post, mask_post = compute_nbr_and_mask(post_items)

valid_2021 = mask_pre & mask_post & forest_mask
dnbr_2021 = (nbr_pre - nbr_post).where(valid_2021)

bounds = [-0.25, -0.1, 0.1, 0.27, 0.44, 0.66]
severity_labels = [
    "Yüksek yeniden büyüme", "Düşük yeniden büyüme", "Yanmamış",
    "Düşük şiddet", "Orta-düşük şiddet", "Orta-yüksek şiddet", "Yüksek şiddet",
]
severity_class = np.digitize(dnbr_2021.values, bounds)
severity_class = np.where(np.isnan(dnbr_2021.values), -1, severity_class)
burned_indices = [3, 4, 5, 6]

print("--- En güncel yıl (2026 Ağustos) ---")
best_date, items = find_best_august_date(2026)
print(f"  Seçilen tarih: {best_date}")
nbr_2026, mask_2026 = compute_nbr_and_mask(items)
nbr_2026 = nbr_2026.rio.reproject_match(nbr_pre)
mask_2026 = mask_2026.astype("uint8").rio.reproject_match(nbr_pre) > 0
valid_2026 = (mask_2026 & forest_mask).values

recovery_amount = (nbr_2026 - nbr_post).values
baseline_vigor = nbr_pre.values

overall_valid = valid_2021 & valid_2026 & np.isin(severity_class, burned_indices)

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
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/recovery_vs_baseline_vigor.png", dpi=150)

print("\nSınıf içi korelasyon (yangın öncesi NBR vs. toparlanma miktarı):")
for label, r, p, n in results:
    print(f"  {label:25s}: r={r:.3f}  p={p:.4f}  n={n}")
print("\nKaydedildi: outputs/recovery_vs_baseline_vigor.png")

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
    "outputs/recovery_dataset.npz",
    baseline_vigor=baseline_vigor[final_valid],
    dnbr_value=dnbr_2021.values[final_valid],
    severity_class=severity_class[final_valid],
    nbr_2026=nbr_2026.values[final_valid],
    elevation=elevation[final_valid],
    slope_deg=slope_deg[final_valid],
    ns_gradient=ns_gradient[final_valid],
    x_coord=xx[final_valid],
    y_coord=yy[final_valid],
)
print("Kaydedildi: outputs/recovery_dataset.npz (model için, koordinatlar dahil)")