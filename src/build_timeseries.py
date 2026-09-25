import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from common import BBOX_4326, get_catalog, get_sentinel2_items, compute_nbr_and_mask, get_forest_mask


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
    return best_date, by_date[best_date]


print("=== Referans: yangın öncesi (2021-07-20) ===")
pre_items = get_sentinel2_items("2021-07-20")
nbr_pre, mask_pre = compute_nbr_and_mask(pre_items)
forest_mask = get_forest_mask(target_grid=nbr_pre)

print("=== Referans: yangın hemen sonrası (2021-08-14) ===")
post_items = get_sentinel2_items("2021-08-14")
nbr_post, mask_post = compute_nbr_and_mask(post_items)

valid_mask_2021 = mask_pre & mask_post & forest_mask
dnbr_2021 = (nbr_pre - nbr_post).where(valid_mask_2021)

bounds = [-0.25, -0.1, 0.1, 0.27, 0.44, 0.66]
severity_labels = [
    "Yüksek yeniden büyüme", "Düşük yeniden büyüme", "Yanmamış",
    "Düşük şiddet", "Orta-düşük şiddet", "Orta-yüksek şiddet", "Yüksek şiddet",
]
severity_class = np.digitize(dnbr_2021.values, bounds)
severity_class = np.where(np.isnan(dnbr_2021.values), -1, severity_class)
burned_indices = [3, 4, 5, 6]  # sadece gerçekten yanmış sınıfları takip ediyoruz

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
    nbr_year = nbr_year.rio.reproject_match(nbr_pre)
    mask_year = mask_year.astype("uint8").rio.reproject_match(nbr_pre) > 0
    valid_year = (mask_year & forest_mask).values
    mean_nbr_per_class(nbr_year, valid_year, str(year))

df = pd.DataFrame(records)
import os
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