import os
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"

import time
from collections import defaultdict

import numpy as np
import rioxarray
from rioxarray.merge import merge_arrays
from pyproj import Transformer
from rasterio.enums import Resampling
import pystac_client
import planetary_computer

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
BBOX_4326 = [30.95, 36.65, 31.75, 37.20]

BAD_SCL_CLASSES = [0, 1, 2, 3, 6, 8, 9, 10, 11]
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


def deduplicate_items_by_tile(items):
    """Aynı tarih+tile için birden fazla işlenmiş sürüm varsa (ESA'nın zaman
    zaman yaptığı yeniden işleme kampanyaları nedeniyle), en güncel sürümü tutar."""
    by_tile = {}
    for it in items:
        tile = it.properties.get("s2:mgrs_tile") or it.id.split("_")[-2]
        existing = by_tile.get(tile)
        if existing is None or it.id > existing.id:
            by_tile[tile] = it
    return list(by_tile.values())


def get_sentinel2_items(date_str, bbox=BBOX_4326):
    search = get_catalog().search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_str}T00:00:00Z/{date_str}T23:59:59Z",
    )
    items = list(search.items())
    return deduplicate_items_by_tile(items)


def get_dates_in_window(year, start_mmdd="07-15", end_mmdd="09-15", bbox=BBOX_4326, cloud_thresh=20):
    """Verilen yıl ve pencerede, bulutluluk eşiğinin altındaki tüm benzersiz
    tarihleri (ve o tarihe ait item'ları) döndürür."""
    search = get_catalog().search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-{start_mmdd}/{year}-{end_mmdd}",
        query={"eo:cloud_cover": {"lt": cloud_thresh}},
    )
    items = list(search.items())
    by_date = defaultdict(list)
    for it in items:
        by_date[it.datetime.date()].append(it)
    return sorted((d, deduplicate_items_by_tile(its)) for d, its in by_date.items())


def find_best_august_date(year, cloud_thresh=20):
    search = get_catalog().search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX_4326,
        datetime=f"{year}-08-01/{year}-08-31",
        query={"eo:cloud_cover": {"lt": cloud_thresh}},
    )
    items = list(search.items())
    if not items:
        return None, []
    by_date = defaultdict(list)
    for it in items:
        by_date[it.datetime.date()].append(it)
    best_date = min(by_date, key=lambda d: min(i.properties["eo:cloud_cover"] for i in by_date[d]))
    return best_date, deduplicate_items_by_tile(by_date[best_date])


def read_mosaic_band(items, band, bbox=BBOX_4326, max_retries=3, retry_delay=3):
    arrays = []
    for it in items:
        print(f"  {band} okunuyor: {it.id}")
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                href = planetary_computer.sign(it.assets[band].href)  # TAZE imza, her okumadan hemen önce
                da = rioxarray.open_rasterio(href, masked=True).squeeze()
                da = clip_to_bbox(da, bbox)
                da = da.load()
                arrays.append(da)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                print(f"    [uyarı] okuma hatası (deneme {attempt}/{max_retries}): {type(e).__name__}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        if last_exc is not None:
            raise last_exc
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
    worldcover = read_mosaic_band(items, "map", bbox)
    worldcover = worldcover.rio.reproject_match(target_grid)
    return worldcover.isin(FOREST_CLASSES)


def get_terrain_features(target_grid, bbox=BBOX_4326):
    print("--- Copernicus DEM (yükseklik/eğim/bakı) ---")
    search = get_catalog().search(collections=["cop-dem-glo-30"], bbox=bbox)
    items = list(search.items())
    print(f"  {len(items)} DEM karosu bulundu")
    dem = read_mosaic_band(items, "data", bbox)
    dem = dem.rio.reproject_match(target_grid, resampling=Resampling.bilinear)

    elevation = dem.values.astype("float32")
    pixel_size = abs(target_grid.rio.resolution()[0])
    dz_dy = -np.gradient(elevation, axis=0) / pixel_size
    dz_dx = np.gradient(elevation, axis=1) / pixel_size
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    return elevation, slope_deg, dz_dy