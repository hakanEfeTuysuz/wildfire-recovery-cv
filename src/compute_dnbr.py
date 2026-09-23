import numpy as np
import matplotlib.pyplot as plt
from common import BBOX_4326, get_sentinel2_items, compute_nbr_and_mask, get_forest_mask

PRE_FIRE_DATE = "2021-07-20"
POST_FIRE_DATE = "2021-08-14"

print("--- Yangın öncesi ---")
pre_items = get_sentinel2_items(PRE_FIRE_DATE)
nbr_pre, mask_pre = compute_nbr_and_mask(pre_items)

print("--- Yangın sonrası ---")
post_items = get_sentinel2_items(POST_FIRE_DATE)
nbr_post, mask_post = compute_nbr_and_mask(post_items)

forest_mask = get_forest_mask(target_grid=nbr_pre)

valid_mask = mask_pre & mask_post & forest_mask
valid_ratio = 100 * float(valid_mask.sum()) / valid_mask.size
print(f"Geçerli (orman+maki + bulutsuz/susuz) piksel oranı: %{valid_ratio:.1f}")

dnbr = (nbr_pre - nbr_post).where(valid_mask)
print("dNBR min/max (geçerli pikseller):", float(dnbr.min()), float(dnbr.max()))

import os
os.makedirs("outputs", exist_ok=True)
dnbr.rio.write_nodata(np.nan, inplace=True)
dnbr.rio.to_raster("outputs/dnbr_manavgat_2021.tif")

fig, ax = plt.subplots(figsize=(8, 8))
dnbr.plot(ax=ax, cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
ax.set_title(f"dNBR (orman+maki): {PRE_FIRE_DATE} vs {POST_FIRE_DATE}")
plt.savefig("outputs/dnbr_preview.png", dpi=150, bbox_inches="tight")
print("Kaydedildi: outputs/dnbr_manavgat_2021.tif ve outputs/dnbr_preview.png")