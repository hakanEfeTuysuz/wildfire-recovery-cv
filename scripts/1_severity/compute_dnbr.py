"""Yangın öncesi/sonrası dNBR haritası → GeoTIFF (data/interim) + önizleme PNG."""
import numpy as np
import matplotlib.pyplot as plt
from wildfire.config import DNBR_TIF, FIGURES_DIR, PRE_FIRE_DATE, POST_FIRE_DATE
from wildfire.imagery import load_fire_reference

ref = load_fire_reference()

valid_ratio = 100 * float(ref.valid.sum()) / ref.valid.size
print(f"Geçerli (orman+maki + bulutsuz/susuz) piksel oranı: %{valid_ratio:.1f}")

dnbr = ref.dnbr
print("dNBR min/max (geçerli pikseller):", float(dnbr.min()), float(dnbr.max()))

dnbr.rio.write_nodata(np.nan, inplace=True)
dnbr.rio.to_raster(DNBR_TIF)

fig, ax = plt.subplots(figsize=(8, 8))
dnbr.plot(ax=ax, cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
ax.set_title(f"dNBR (orman+maki): {PRE_FIRE_DATE} vs {POST_FIRE_DATE}")
out_png = FIGURES_DIR / "dnbr_preview.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Kaydedildi: {DNBR_TIF} ve {out_png}")
