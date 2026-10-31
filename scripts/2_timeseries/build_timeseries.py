"""Yıllık (her yılın en az bulutlu Ağustos görüntüsü) şiddet sınıfı bazında NBR zaman serisi."""
import pandas as pd
import matplotlib.pyplot as plt
from wildfire.config import FIGURES_DIR, TABLES_DIR, TARGET_YEAR
from wildfire.imagery import compute_nbr_and_mask, find_best_august_date, align_to_reference, load_fire_reference
from wildfire.severity import SEVERITY_LABELS, BURNED_CLASSES

ref = load_fire_reference()
nbr_pre, mask_pre, forest_mask = ref.nbr_pre, ref.mask_pre, ref.forest_mask
severity_labels = SEVERITY_LABELS
severity_class = ref.severity_class
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
mean_nbr_per_class(ref.nbr_post, ref.valid.values, "2021_sonrasi")

for year in range(2022, TARGET_YEAR + 1):
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
out_csv = TABLES_DIR / "recovery_timeseries.csv"
df.to_csv(out_csv, index=False)
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
out_png = FIGURES_DIR / "recovery_timeseries.png"
plt.savefig(out_png, dpi=150)
print(f"Kaydedildi: {out_csv} ve {out_png}")
