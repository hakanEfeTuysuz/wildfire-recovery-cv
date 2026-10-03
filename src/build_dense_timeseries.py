import os
from datetime import date
import numpy as np
import pandas as pd
from common import get_sentinel2_items, compute_nbr_and_mask, get_forest_mask, get_dates_in_window

PRE_FIRE_DATE = "2021-07-20"
POST_FIRE_DATE = "2021-08-14"
T0_DATE = date(2021, 8, 14)
OUT_PATH = "outputs/recovery_timeseries_dense.csv"

bounds = [-0.25, -0.1, 0.1, 0.27, 0.44, 0.66]
severity_labels = [
    "Yüksek yeniden büyüme", "Düşük yeniden büyüme", "Yanmamış",
    "Düşük şiddet", "Orta-düşük şiddet", "Orta-yüksek şiddet", "Yüksek şiddet",
]
burned_indices = [3, 4, 5, 6]

os.makedirs("outputs", exist_ok=True)

if os.path.exists(OUT_PATH):
    existing = pd.read_csv(OUT_PATH)
    done_dates = set(existing.dropna(subset=["ortalama_nbr"])["tarih"].astype(str).unique())
    bad_dates = set(existing["tarih"].astype(str).unique()) - done_dates
    if bad_dates:
        print(f"{len(bad_dates)} tarih önceki çalıştırmada bozuk (NaN) veri bırakmış, yeniden denenecek")
else:
    done_dates = set()


def append_rows(rows):
    if not rows:
        return
    df_chunk = pd.DataFrame(rows)
    write_header = not os.path.exists(OUT_PATH)
    df_chunk.to_csv(OUT_PATH, mode="a", header=write_header, index=False)


print("--- Yangın öncesi (referans) ---")
pre_items = get_sentinel2_items(PRE_FIRE_DATE)
nbr_pre, mask_pre = compute_nbr_and_mask(pre_items)
forest_mask = get_forest_mask(target_grid=nbr_pre)

print("--- Yangın hemen sonrası (t=0, referans) ---")
post_items = get_sentinel2_items(POST_FIRE_DATE)
nbr_post, mask_post = compute_nbr_and_mask(post_items)
valid_2021 = mask_pre & mask_post & forest_mask
dnbr_2021 = (nbr_pre - nbr_post).where(valid_2021)

severity_class = np.digitize(dnbr_2021.values, bounds)
severity_class = np.where(np.isnan(dnbr_2021.values), -1, severity_class)


MIN_VALID_PIXELS = 50

def build_rows(mean_arr, valid_2d, tarih, t_yil):
    rows = []
    for idx in burned_indices:
        class_mask = (severity_class == idx) & valid_2d
        values = mean_arr[class_mask]
        values = values[~np.isnan(values)]
        if len(values) < MIN_VALID_PIXELS:
            continue
        mean_val = float(values.mean())
        rows.append({"tarih": tarih, "t_yil": t_yil,
                      "siddet_sinifi": severity_labels[idx], "ortalama_nbr": mean_val})
    return rows


if str(T0_DATE) not in done_dates:
    append_rows(build_rows(nbr_post.values, valid_2021.values, str(T0_DATE), 0.0))
    done_dates.add(str(T0_DATE))
if PRE_FIRE_DATE not in done_dates:
    append_rows(build_rows(nbr_pre.values, (mask_pre & forest_mask).values, PRE_FIRE_DATE, np.nan))
    done_dates.add(PRE_FIRE_DATE)

for year in range(2021, 2027):
    start = "08-15" if year == 2021 else "07-15"
    print(f"--- {year} yaz penceresi ({start} - 09-15) taranıyor ---")
    date_items = get_dates_in_window(year, start_mmdd=start, end_mmdd="09-15")
    print(f"  {len(date_items)} uygun tarih bulundu")
    for d, items in date_items:
        tarih_str = str(d)
        if tarih_str in done_dates:
            print(f"  {d} zaten işlenmiş, atlanıyor")
            continue
        t_yil = (d - T0_DATE).days / 365.25
        if t_yil <= 0:
            continue
        print(f"  {d} işleniyor (t={t_yil:.2f} yıl, {len(items)} görüntü)")
        try:
            nbr_d, mask_d = compute_nbr_and_mask(items)
            nbr_d = nbr_d.rio.reproject_match(nbr_pre)
            mask_d = mask_d.astype("uint8").rio.reproject_match(nbr_pre) > 0
            valid_d = (mask_d & forest_mask).values
            append_rows(build_rows(nbr_d.values, valid_d, tarih_str, t_yil))
            done_dates.add(tarih_str)
        except Exception as e:
            print(f"  [UYARI] {d} işlenemedi, atlanıyor: {type(e).__name__}: {e}")
            continue

print(f"\nBitti. Tüm sonuçlar {OUT_PATH} dosyasında birikti.")
final_df = pd.read_csv(OUT_PATH)
print(final_df.groupby("siddet_sinifi")["t_yil"].count())