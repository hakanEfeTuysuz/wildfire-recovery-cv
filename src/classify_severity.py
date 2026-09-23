import numpy as np
import rioxarray
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

dnbr = rioxarray.open_rasterio("outputs/dnbr_manavgat_2021.tif", masked=True).squeeze()
values = dnbr.values

bounds = [-0.25, -0.1, 0.1, 0.27, 0.44, 0.66]
labels = [
    "Yüksek yeniden büyüme", "Düşük yeniden büyüme", "Yanmamış",
    "Düşük şiddet", "Orta-düşük şiddet", "Orta-yüksek şiddet", "Yüksek şiddet",
]
colors = ["#0b6623", "#66bb6a", "#fff9c4", "#ffd54f", "#ff8a65", "#e53935", "#7b0000"]

valid = ~np.isnan(values)
severity_class = np.full(values.shape, -1, dtype=int)  # -1 = orman dışı / maskeli
severity_class[valid] = np.digitize(values[valid], bounds)

pixel_area_ha = (10 * 10) / 10000
print("Sınıf bazlı alan (sadece orman pikselleri):")
total_burned_ha = 0
for i, label in enumerate(labels):
    area_ha = int(np.sum(severity_class == i)) * pixel_area_ha
    print(f"  {label:25s}: {area_ha:>10,.0f} ha")
    if i >= 3:
        total_burned_ha += area_ha
print(f"\nToplam yanan orman alanı (düşük şiddet ve üzeri): {total_burned_ha:,.0f} ha")

cmap = ListedColormap(["#e0e0e0"] + colors)  # ilk renk: maskeli alan (gri)
norm = BoundaryNorm(range(-1, len(labels) + 1), cmap.N)

fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(severity_class, cmap=cmap, norm=norm)
cbar = fig.colorbar(im, ticks=[i + 0.5 for i in range(-1, len(labels))])
cbar.ax.set_yticklabels(["Orman dışı/maskeli"] + labels)
ax.set_title("Yanık Şiddeti Sınıflandırması (sadece orman) — Manavgat 2021")
ax.axis("off")
plt.savefig("outputs/severity_classified.png", dpi=150, bbox_inches="tight")
print("Kaydedildi: outputs/severity_classified.png")