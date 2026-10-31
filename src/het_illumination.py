import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (rio erişimcisi için)
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import get_terrain_features
from het_core import load_dataset, condition_matrix, SpatialFolds, MAX_TRAIN, SEED

CELL = 10
# Sentinel-2 geçişi yaklaşık 08:35 UTC (Ağustos, ~37°K, 31.4°D): güneş güneydoğuda
SUN_AZ_DEG = 131.0
SUN_EL_DEG = 63.0
BLOCKS = (1000, 3000)
SHALLOW_ITER = 100
W_FINE, W_MID, W_WIDE = 5, 101, 301

d = load_dataset()
x_c, y_c = d["x_coord"], d["y_coord"]
y = d["nbr_2026"].astype("float64")
X = condition_matrix(d)

# --- Hedef ızgarayı koordinatlardan kur, DEM'i bu ızgaraya oturt ---
xmin, xmax, ymin, ymax = x_c.min(), x_c.max(), y_c.min(), y_c.max()
xs = xmin + CELL * np.arange(int(np.rint((xmax - xmin) / CELL)) + 1)
ys = ymax - CELL * np.arange(int(np.rint((ymax - ymin) / CELL)) + 1)
H, W = len(ys), len(xs)
grid = xr.DataArray(np.zeros((H, W), dtype="float32"), coords={"y": ys, "x": xs}, dims=("y", "x"))
grid = grid.rio.write_crs("EPSG:32636")   # WGS 84 / UTM 36N (T36S karoları)

elevation, _, _ = get_terrain_features(grid)
dz_dx = np.gradient(elevation, axis=1) / CELL          # doğuya doğru yükseliş
dz_dy = -np.gradient(elevation, axis=0) / CELL         # kuzeye doğru yükseliş
az, el = np.deg2rad(SUN_AZ_DEG), np.deg2rad(SUN_EL_DEG)
hillshade = (-dz_dx * np.cos(el) * np.sin(az) - dz_dy * np.cos(el) * np.cos(az) + np.sin(el)) / np.sqrt(1 + dz_dx ** 2 + dz_dy ** 2)

ix = np.rint((x_c - xmin) / CELL).astype(np.int64)
iy = np.rint((ymax - y_c) / CELL).astype(np.int64)
ew = dz_dx[iy, ix].astype("float32")
hs = hillshade[iy, ix].astype("float32")
diff = np.abs(elevation[iy, ix] - d["elevation"])
print(f"Hizalama kontrolü: DEM yüksekliği ile veri setindeki yükseklik farkı medyan {np.nanmedian(diff):.2f} m, "
      f"en büyük {np.nanmax(diff):.1f} m (medyan ~0 olmalı)")
del dz_dx, dz_dy, hillshade, elevation

X2 = np.column_stack([X, ew, hs]).astype("float32")


def mk_linear():
    return make_pipeline(StandardScaler(), LinearRegression())


def mk_shallow():
    return HistGradientBoostingRegressor(max_iter=SHALLOW_ITER, learning_rate=0.1, max_depth=3, random_state=SEED)


def oof(folds, Xm, make_model):
    rng = np.random.default_rng(SEED)
    pred = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    for i in range(folds.k):
        idx = np.flatnonzero(folds.train_masks[i])
        if len(idx) > MAX_TRAIN:
            idx = rng.choice(idx, MAX_TRAIN, replace=False)
        m = make_model().fit(Xm[idx], y[idx])
        test = folds.fold == i
        pred[test] = m.predict(Xm[test])
        base[test] = y[idx].mean()
    return pred, base


def smooth(g, size):
    ok = np.isfinite(g)
    num = ndimage.uniform_filter(np.where(ok, g, 0).astype("float32"), size=size, mode="constant")
    den = ndimage.uniform_filter(ok.astype("float32"), size=size, mode="constant")
    return np.where(den > 0.2, num / np.maximum(den, 1e-6), np.nan).astype("float32")


def scale_shares(resid):
    g = np.full((H, W), np.nan, dtype="float32")
    g[iy, ix] = resid
    s1, s2, s3 = smooth(g, W_FINE), smooth(g, W_MID), smooth(g, W_WIDE)
    comps = {
        "< 50 m": (g - s1)[iy, ix],
        "50 m - 1 km": (s1 - s2)[iy, ix],
        "1 - 3 km": (s2 - s3)[iy, ix],
        "> 3 km": s3[iy, ix],
    }
    del g, s1, s2, s3
    ok = np.ones(len(resid), dtype=bool)
    for c in comps.values():
        ok &= np.isfinite(c)
    tot = float(np.var(resid[ok]))
    return {k: float(np.var(v[ok])) / tot for k, v in comps.items()}


sst = float(np.sum((y - y.mean()) ** 2))
rows = []
resid_by_setup = {}
for block in BLOCKS:
    folds = SpatialFolds(x_c, y_c, block=block)
    for mname, mk in (("Doğrusal", mk_linear), ("Sığ ağaç", mk_shallow)):
        for setup, Xm in (("5 koşul", X), ("5 koşul + doğu-batı eğim + gölgeleme", X2)):
            print(f"  blok {block} m, {mname}, {setup} ...")
            pred, base = oof(folds, Xm, mk)
            sse = float(np.sum((y - pred) ** 2))
            rows.append({
                "blok (m)": block, "model": mname, "koşul kümesi": setup,
                "R²": round(1 - sse / sst, 3),
                "beceri skoru": round(1 - sse / float(np.sum((y - base) ** 2)), 3),
            })
            if block == 1000 and mname == "Doğrusal":
                resid_by_setup[setup] = y - pred
tab = pd.DataFrame(rows)
print("\n=== 1) Aydınlanma değişkenleri modele katkı sağlıyor mu? ===")
print(tab.to_string(index=False))
tab.to_csv("outputs/het_illumination_skill.csv", index=False)

r0 = resid_by_setup["5 koşul"]
print("\n=== 2) Mevcut artığın (5 koşullu, doğrusal, 1 km) aydınlanma değişkenleriyle ilişkisi ===")
print(f"  r(artık, doğu-batı eğim) = {np.corrcoef(r0, ew)[0, 1]:+.3f}")
print(f"  r(artık, gölgeleme)      = {np.corrcoef(r0, hs)[0, 1]:+.3f}")

print("\n=== 3) Artık varyansının ölçek payları: aydınlanma değişkenleri eklenmeden / eklendikten sonra ===")
s_before = scale_shares(resid_by_setup["5 koşul"])
s_after = scale_shares(resid_by_setup["5 koşul + doğu-batı eğim + gölgeleme"])
v0 = float(np.var(resid_by_setup["5 koşul"]))
v1 = float(np.var(resid_by_setup["5 koşul + doğu-batı eğim + gölgeleme"]))
out = pd.DataFrame({
    "ölçek": list(s_before.keys()),
    "payı x toplam artık varyansı (önce)": [round(s_before[k] * v0 * 1e4, 2) for k in s_before],
    "payı x toplam artık varyansı (sonra)": [round(s_after[k] * v1 * 1e4, 2) for k in s_after],
})
print("(birim: 1e-4 NBR²; sütunlar mutlak varyans katkısıdır, doğrudan karşılaştırılabilir)")
print(out.to_string(index=False))
out.to_csv("outputs/het_illumination_scales.csv", index=False)