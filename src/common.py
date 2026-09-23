import os
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"

import rioxarray
from rioxarray.merge import merge_arrays
from pyproj import Transformer
import pystac_client
import planetary_computer

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX_4326 = [30.95, 36.65, 31.75, 37.20]

# SCL: 0 nodata, 1 saturated, 2 dark area, 3 cloud shadow, 6 water, 8/9/10 cloud, 11 snow
BAD_SCL_CLASSES = [0, 1, 2, 3, 6, 8, 9, 10, 11]
# ESA WorldCover: 10=Tree cover, 20=Shrubland (maki)
FOREST_CLASSES = [10, 20]

_catalog = None


def get_catalog():
    global _catalog
    if _catalog is None:
        _catalog = pystac_client.Client.open(
            CATALOG_URL, modifier=planetary_computer.sign_inplace
        )
    return _catalog


def clip_to_bbox(da, bbox=BBOX_4326):
    transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])
    return da.rio.clip_box(minx, miny, maxx, maxy)


def get_sentinel2_items(date_str, bbox=BBOX_4326):
    search = get_catalog().search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_str}T00:00:00Z/{date_str}T23:59:59Z",
    )
    return list(search.items())


def read_mosaic_band(items, band, bbox=BBOX_4326):
    arrays = []
    for it in items:
        print(f"  {band} okunuyor: {it.id}")
        da = rioxarray.open_rasterio(it.assets[band].href, masked=True).squeeze()
        da = clip_to_bbox(da, bbox)
        arrays.append(da)
    return arrays[0] if len(arrays) == 1 else merge_arrays(arrays)


def compute_nbr_and_mask(items, bbox=BBOX_4326):
    nir = read_mosaic_band(items, "B08", bbox).astype("float32")
    swir2 = read_mosaic_band(items, "B12", bbox).astype("float32")
    swir2 = swir2.rio.reproject_match(nir)
    scl = read_mosaic_band(items, "SCL", bbox)
    scl = scl.rio.reproject_match(nir)

    nbr = (nir - swir2) / (nir + swir2 + 1e-6)
    good_mask = ~scl.isin(BAD_SCL_CLASSES)
    return nbr, good_mask


def get_forest_mask(target_grid, bbox=BBOX_4326):
    print("--- ESA WorldCover (orman + maki maskesi) ---")
    search = get_catalog().search(
        collections=["esa-worldcover"], bbox=bbox, datetime="2020-01-01/2020-12-31"
    )
    items = list(search.items())
    arrays = []
    for it in items:
        print(f"  map okunuyor: {it.id}")
        da = rioxarray.open_rasterio(it.assets["map"].href).squeeze()
        da = clip_to_bbox(da, bbox)
        arrays.append(da)
    worldcover = arrays[0] if len(arrays) == 1 else merge_arrays(arrays)
    worldcover = worldcover.rio.reproject_match(target_grid)
    return worldcover.isin(FOREST_CLASSES)