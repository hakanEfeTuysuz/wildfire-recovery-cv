import os
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"

import numpy as np
import rioxarray
from rioxarray.merge import merge_arrays
from pyproj import Transformer
import matplotlib.pyplot as plt
import pystac_client
import planetary_computer

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

BBOX_4326 = [30.95, 36.65, 31.75, 37.20]
PRE_FIRE_DATE = "2021-07-20"
POST_FIRE_DATE = "2021-08-14"

# SCL sınıfları: 0 nodata, 1 saturated, 2 dark area, 3 cloud shadow,
# 6 water, 8/9/10 cloud, 11 snow — hepsini eliyoruz
BAD_SCL_CLASSES = [0, 1, 2, 3, 6, 8, 9, 10, 11]
FOREST_CLASS = 10  # ESA WorldCover: Tree cover


def get_items(date_str):
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX_4326,
        datetime=f"{date_str}T00:00:00Z/{date_str}T23:59:59Z",
    )
    return list(search.items())


def clip_to_bbox(da):
    transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
    minx, miny = transformer.transform(BBOX_4326[0], BBOX_4326[1])
    maxx, maxy = transformer.transform(BBOX_4326[2], BBOX_4326[3])
    return da.rio.clip_box(minx, miny, maxx, maxy)


def read_mosaic_band(items, band):
    arrays = []
    for it in items:
        print(f"  {band} okunuyor: {it.id}")
        da = rioxarray.open_rasterio(it.assets[band].href, masked=True).squeeze()
        da = clip_to_bbox(da)
        arrays.append(da)
    return arrays[0] if len(arrays) == 1 else merge_arrays(arrays)


def compute_nbr_and_mask(items):
    nir = read_mosaic_band(items, "B08").astype("float32")
    swir2 = read_mosaic_band(items, "B12").astype("float32")
    swir2 = swir2.rio.reproject_match(nir)
    scl = read_mosaic_band(items, "SCL")
    scl = scl.rio.reproject_match(nir)

    nbr = (nir - swir2) / (nir + swir2 + 1e-6)
    good_mask = ~scl.isin(BAD_SCL_CLASSES)
    return nbr, good_mask


def get_forest_mask(target_grid):
    print("--- ESA WorldCover (orman maskesi) ---")
    search = catalog.search(
        collections=["esa-worldcover"],
        bbox=BBOX_4326,
        datetime="2020-01-01/2020-12-31",
    )
    items = list(search.items())
    print(f"  {len(items)} WorldCover karosu bulundu")
    arrays = []
    for it in items:
        print(f"  map okunuyor: {it.id}, asset anahtarları: {list(it.assets.keys())}")
        da = rioxarray.open_rasterio(it.assets["map"].href).squeeze()
        da = clip_to_bbox(da)
        arrays.append(da)
    worldcover = arrays[0] if len(arrays) == 1 else merge_arrays(arrays)
    worldcover = worldcover.rio.reproject_match(target_grid)
    return worldcover == FOREST_CLASS


print("--- Yangın öncesi ---")
pre_items = get_items(PRE_FIRE_DATE)
nbr_pre, mask_pre = compute_nbr_and_mask(pre_items)

print("--- Yangın sonrası ---")
post_items = get_items(POST_FIRE_DATE)
nbr_post, mask_post = compute_nbr_and_mask(post_items)

forest_mask = get_forest_mask(target_grid=nbr_pre)

valid_mask = mask_pre & mask_post & forest_mask
valid_ratio = 100 * float(valid_mask.sum()) / valid_mask.size
print(f"Geçerli (orman + bulutsuz/susuz) piksel oranı: %{valid_ratio:.1f}")

dnbr = (nbr_pre - nbr_post).where(valid_mask)
print("dNBR min/max (geçerli pikseller):", float(dnbr.min()), float(dnbr.max()))

os.makedirs("outputs", exist_ok=True)
dnbr.rio.write_nodata(np.nan, inplace=True)
dnbr.rio.to_raster("outputs/dnbr_manavgat_2021.tif")

fig, ax = plt.subplots(figsize=(8, 8))
dnbr.plot(ax=ax, cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
ax.set_title(f"dNBR (sadece orman): {PRE_FIRE_DATE} vs {POST_FIRE_DATE}")
plt.savefig("outputs/dnbr_preview.png", dpi=150, bbox_inches="tight")
print("Kaydedildi: outputs/dnbr_manavgat_2021.tif ve outputs/dnbr_preview.png")