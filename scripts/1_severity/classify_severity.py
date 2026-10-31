"""dNBR GeoTIFF'ini USGS şiddet sınıflarına ayırır; sınıf alanları ve harita."""
import numpy as np
import rioxarray
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from wildfire.config import DNBR_TIF, FIGURES_DIR
from wildfire.severity import classify_severity, SEVERITY_LABELS, SEVERITY_COLORS

dnbr = rioxarray.open_rasterio(DNBR_TIF, masked=True).squeeze()
values = dnbr.values

labels = SEVERITY_LABELS
colors = SEVERITY_COLORS

severity_class = classify_severity(values)  # -1 = orman dışı / maskeli

res_x, res_y = dnbr.rio.resolution()
pixel_area_ha = abs(res_x * res_y) / 10000
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
out_png = FIGURES_DIR / "severity_classified.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Kaydedildi: {out_png}")
