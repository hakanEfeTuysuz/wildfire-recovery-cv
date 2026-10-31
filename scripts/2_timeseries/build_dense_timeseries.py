"""Yoğun zaman serisi: her yılın 15 Temmuz - 15 Eylül penceresindeki tüm bulutsuz
görüntüler. Uzun sürer; kesilirse kaldığı tarihten devam eder."""
import os
from datetime import date
import numpy as np
import pandas as pd
from wildfire.config import INTERIM_DIR, TABLES_DIR, PRE_FIRE_DATE, POST_FIRE_DATE, TARGET_YEAR, PIPELINE_VERSION
from wildfire.imagery import compute_nbr_and_mask, get_dates_in_window, align_to_reference, load_fire_reference
from wildfire.severity import SEVERITY_LABELS, BURNED_CLASSES

T0_DATE = date.fromisoformat(POST_FIRE_DATE)
OUT_PATH = TABLES_DIR / "recovery_timeseries_dense.csv"

severity_labels = SEVERITY_LABELS
burned_indices = BURNED_CLASSES

if os.path.exists(OUT_PATH):
    existing = pd.read_csv(OUT_PATH)
    if "surum" not in existing.columns or (existing["surum"] != PIPELINE_VERSION).any():
        # Eski sürümle (ör. Sentinel-2 ofset düzeltmesi öncesi) üretilmiş satırlar
        # yeni satırlarla karıştırılmamalı: dosya yedeklenip baştan başlanır.
        backup = INTERIM_DIR / f"{OUT_PATH.stem}_eski_{date.today()}.csv"
        os.replace(OUT_PATH, backup)
        print(f"Mevcut dosya eski bir işleme sürümüne ait; {backup} olarak yedeklendi, baştan başlanıyor")
        existing = pd.DataFrame(columns=["tarih", "ortalama_nbr"])
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


ref = load_fire_reference()
nbr_pre, mask_pre, forest_mask = ref.nbr_pre, ref.mask_pre, ref.forest_mask
severity_class = ref.severity_class


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
                      "siddet_sinifi": severity_labels[idx], "ortalama_nbr": mean_val,
                      "surum": PIPELINE_VERSION})
    return rows


if str(T0_DATE) not in done_dates:
    append_rows(build_rows(ref.nbr_post.values, ref.valid.values, str(T0_DATE), 0.0))
    done_dates.add(str(T0_DATE))
if PRE_FIRE_DATE not in done_dates:
    append_rows(build_rows(nbr_pre.values, (mask_pre & forest_mask).values, PRE_FIRE_DATE, np.nan))
    done_dates.add(PRE_FIRE_DATE)

for year in range(T0_DATE.year, TARGET_YEAR + 1):
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
            nbr_d, mask_d = align_to_reference(nbr_d, mask_d, nbr_pre)
            valid_d = (mask_d & forest_mask).values
            append_rows(build_rows(nbr_d.values, valid_d, tarih_str, t_yil))
            done_dates.add(tarih_str)
        except Exception as e:
            print(f"  [UYARI] {d} işlenemedi, atlanıyor: {type(e).__name__}: {e}")
            continue

print(f"\nBitti. Tüm sonuçlar {OUT_PATH} dosyasında birikti.")
final_df = pd.read_csv(OUT_PATH)
print(final_df.groupby("siddet_sinifi")["t_yil"].count())