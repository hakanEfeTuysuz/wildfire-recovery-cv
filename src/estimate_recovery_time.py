import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("outputs/recovery_timeseries.csv")

period_to_t = {"2021_sonrasi": 0, "2022": 1, "2023": 2, "2024": 3, "2025": 4, "2026": 5}
severity_order = ["Düşük şiddet", "Orta-düşük şiddet", "Orta-yüksek şiddet", "Yüksek şiddet"]
colors = {"Düşük şiddet": "tab:blue", "Orta-düşük şiddet": "tab:orange",
          "Orta-yüksek şiddet": "tab:green", "Yüksek şiddet": "tab:red"}

fig, ax = plt.subplots(figsize=(10, 7))
print(f"{'Sınıf':22s} {'Öncesi NBR':>10s} {'Tavan (A)':>10s} {'tau (yıl)':>10s}  Tahmini toparlanma")

for label in severity_order:
    sub = df[df["siddet_sinifi"] == label]
    nbr_pre = float(sub[sub["donem"] == "2021_oncesi"]["ortalama_nbr"].iloc[0])

    t_vals, y_vals = [], []
    for period, t in period_to_t.items():
        row = sub[sub["donem"] == period]
        if not row.empty:
            t_vals.append(t)
            y_vals.append(float(row["ortalama_nbr"].iloc[0]))
    t_vals = np.array(t_vals)
    y_vals = np.array(y_vals)
    nbr_t0 = y_vals[0]  # t=0 (yangın hemen sonrası) — model bu noktadan geçmek zorunda

    def model(t, A, tau, nbr_t0=nbr_t0):
        return A - (A - nbr_t0) * np.exp(-t / tau)

    try:
        popt, _ = curve_fit(model, t_vals, y_vals, p0=[nbr_pre, 3.0], bounds=([-1, 0.2], [1, 100]))
    except RuntimeError:
        print(f"  {label}: eğri uydurulamadı")
        continue
    A, tau = popt

    if A > nbr_pre:
        t_recover = -tau * np.log((A - nbr_pre) / (A - nbr_t0))
        recover_str = f"~{t_recover:.1f} yıl (yangından itibaren)"
    else:
        recover_str = f"tavan (A={A:.3f}) < yangın öncesi ({nbr_pre:.3f}) → 5 yıllık veriyle tam toparlanma öngörülemiyor"

    print(f"{label:22s} {nbr_pre:10.3f} {A:10.3f} {tau:10.2f}  {recover_str}")

    t_plot = np.linspace(0, 15, 200)
    y_plot = model(t_plot, A, tau)
    ax.scatter(t_vals, y_vals, color=colors[label], label=f"{label} (gözlem)")
    ax.plot(t_plot, y_plot, color=colors[label], linestyle="--", alpha=0.7)
    ax.axhline(nbr_pre, color=colors[label], linestyle=":", alpha=0.35)

ax.axvline(5, color="gray", linewidth=0.8)
ax.set_xlabel("Yangından itibaren geçen yıl")
ax.set_ylabel("Ortalama NBR")
ax.set_title("Toparlanma Eğrileri ve Ekstrapolasyon (üstel doygunluk modeli)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/recovery_curves.png", dpi=150)
print("\nKaydedildi: outputs/recovery_curves.png")
print("\nNOT: tahminler sadece 5 yıllık gözlemden ekstrapole ediliyor — özellikle yüksek")
print("şiddet sınıfında büyük belirsizlik taşır, kesin bir öngörü değil kaba bir tahmindir.")