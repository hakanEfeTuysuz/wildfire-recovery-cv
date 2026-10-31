import os
import pandas as pd
import matplotlib.pyplot as plt
from common import (
    get_sentinel2_items, compute_nbr_and_mask, get_forest_mask, find_best_august_date,
    align_to_reference, classify_severity, SEVERITY_LABELS, BURNED_CLASSES,
)

print("=== Referans: yangın öncesi (2021-07-20) ===")
pre_items = get_sentinel2_items("2021-07-20")
nbr_pre, mask_pre = compute_nbr_and_mask(pre_items)
forest_mask = get_forest_mask(target_grid=nbr_pre)

print("=== Referans: yangın hemen sonrası (2021-08-14) ===")
post_items = get_sentinel2_items("2021-08-14")
nbr_post, mask_post = compute_nbr_and_mask(post_items)
nbr_post, mask_post = align_to_reference(nbr_post, mask_post, nbr_pre)

valid_mask_2021 = mask_pre & mask_post & forest_mask
dnbr_2021 = (nbr_pre - nbr_post).where(valid_mask_2021)

severity_labels = SEVERITY_LABELS
severity_class = classify_severity(dnbr_2021.values)
burned_indices = BURNED_CLASSES  # sadece gerçekten yanmış sınıfları takip ediyoruz

records = []


def mean_nbr_per_class(nbr_array, valid_2d, label):
    for idx in burned_indices:
        class_mask = (severity_class == idx) & valid_2d
        if class_mask.sum() == 0:
            continue
        mean_val = float(nbr_array.values[class_mask].mean())
        records.append({"donem": label, "siddet_sinifi": severity_labels[idx], "ortalama_nbr": mean_val})


mean_nbr_per_class(nbr_pre, (mask_pre & forest_mask).values, "2021_oncesi")
mean_nbr_per_class(nbr_post, valid_mask_2021.values, "2021_sonrasi")

for year in [2022, 2023, 2024, 2025, 2026]:
    print(f"=== {year} Ağustos ===")
    best_date, items = find_best_august_date(year)
    if not items:
        print(f"  {year} için uygun (bulutsuz) görüntü bulunamadı, atlanıyor")
        continue
    print(f"  Seçilen tarih: {best_date}")
    nbr_year, mask_year = compute_nbr_and_mask(items)
    nbr_year, mask_year = align_to_reference(nbr_year, mask_year, nbr_pre)
    valid_year = (mask_year & forest_mask).values
    mean_nbr_per_class(nbr_year, valid_year, str(year))

df = pd.DataFrame(records)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/recovery_timeseries.csv", index=False)
print(df)

fig, ax = plt.subplots(figsize=(9, 6))
for label in severity_labels[3:]:
    sub = df[df["siddet_sinifi"] == label]
    ax.plot(sub["donem"], sub["ortalama_nbr"], marker="o", label=label)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_ylabel("Ortalama NBR")
ax.set_title("Yangın Sonrası Toparlanma — Şiddet Sınıfına Göre NBR Zaman Serisi")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/recovery_timeseries.png", dpi=150)
print("Kaydedildi: outputs/recovery_timeseries.csv ve outputs/recovery_timeseries.png")