import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

df = pd.read_csv("outputs/recovery_timeseries_dense.csv")
df = df.dropna(subset=["ortalama_nbr"])

severity_order = ["Düşük şiddet", "Orta-düşük şiddet", "Orta-yüksek şiddet", "Yüksek şiddet"]
colors = {"Düşük şiddet": "tab:blue", "Orta-düşük şiddet": "tab:orange",
          "Orta-yüksek şiddet": "tab:green", "Yüksek şiddet": "tab:red"}

TAU_BOUNDS = (0.15, 50)
BOUND_TOLERANCE = 0.05


def model(params, t):
    A, tau, N0 = params
    return A - (A - N0) * np.exp(-t / tau)


def residuals(params, t, y):
    return model(params, t) - y


fig, ax = plt.subplots(figsize=(10, 7))
print(f"{'Sınıf':22s} {'n':>4s} {'Öncesi NBR':>10s} {'Tavan (A)':>10s} {'tau (yıl)':>10s}  Tahmini toparlanma")

for label in severity_order:
    sub = df[df["siddet_sinifi"] == label]
    nbr_pre = float(sub[sub["t_yil"].isna()]["ortalama_nbr"].iloc[0])

    fit_sub = sub[sub["t_yil"].notna()].sort_values("t_yil")
    t_vals = fit_sub["t_yil"].values.astype(float)
    y_vals = fit_sub["ortalama_nbr"].values.astype(float)
    nbr_t0 = y_vals[0]

    best_result = None
    for tau0 in [0.3, 0.7, 1.5, 3.0, 6.0, 10.0]:
        p0 = [max(nbr_pre, nbr_t0 + 0.05), tau0, nbr_t0]
        try:
            result = least_squares(
                residuals, p0, args=(t_vals, y_vals),
                bounds=([-1, TAU_BOUNDS[0], -1], [1, TAU_BOUNDS[1], 1]),
                loss="soft_l1", f_scale=0.05,
            )
        except Exception:
            continue
        if best_result is None or result.cost < best_result.cost:
            best_result = result

    if best_result is None:
        print(f"  {label}: eğri uydurulamadı")
        continue
    A, tau, N0 = best_result.x

    tau_unreliable = (
        tau > TAU_BOUNDS[1] * (1 - BOUND_TOLERANCE)
        or tau < TAU_BOUNDS[0] * (1 + BOUND_TOLERANCE)
    )

    if tau_unreliable:
        recover_str = "BELİRLENEMEDİ — veri doygunluk emaresi göstermiyor"
    elif A > nbr_pre:
        t_recover = -tau * np.log((A - nbr_pre) / (A - N0))
        recover_str = f"~{t_recover:.1f} yıl (yangından itibaren)"
        try:
            dof = max(len(t_vals) - 3, 1)
            residual_var = 2 * best_result.cost / dof
            cov = residual_var * np.linalg.inv(best_result.jac.T @ best_result.jac)
            rng = np.random.default_rng(42)
            samples = rng.multivariate_normal([A, tau, N0], cov, size=3000)
            t_samples, never_recovers = [], 0
            for As, taus, N0s in samples:
                if As <= nbr_pre or taus <= 0 or As <= N0s:
                    never_recovers += 1
                    continue
                t_samples.append(-taus * np.log((As - nbr_pre) / (As - N0s)))
            never_recovers_pct = 100 * never_recovers / len(samples)
            if len(t_samples) > 200:
                lo, hi = np.percentile(t_samples, [5, 95])
                recover_str += f"  [%90 aralık: {lo:.0f}-{hi:.0f} yıl]"
            if never_recovers_pct > 15:
                recover_str += f"  ⚠ %{never_recovers_pct:.0f} kırılgan"
        except np.linalg.LinAlgError:
            pass
    else:
        recover_str = f"tavan (A={A:.3f}) < yangın öncesi ({nbr_pre:.3f}) → toparlanma öngörülemiyor"

    print(f"{label:22s} {len(t_vals):4d} {nbr_pre:10.3f} {A:10.3f} {tau:10.2f}  {recover_str}")

    t_max_plot = max(t_vals) if tau_unreliable else 15
    t_plot = np.linspace(0, t_max_plot, 200)
    y_plot = model([A, tau, N0], t_plot)
    ax.scatter(t_vals, y_vals, color=colors[label], s=15, alpha=0.6, label=f"{label} (gözlem, n={len(t_vals)})")
    linestyle = ":" if tau_unreliable else "--"
    ax.plot(t_plot, y_plot, color=colors[label], linestyle=linestyle, alpha=0.8)
    ax.axhline(nbr_pre, color=colors[label], linestyle=":", alpha=0.3)

ax.axvline(5, color="gray", linewidth=0.8)
ax.set_xlabel("Yangından itibaren geçen yıl")
ax.set_ylabel("Ortalama NBR")
ax.set_title("Toparlanma Eğrileri (yoğun zaman serisi)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/recovery_curves.png", dpi=150)
print("\nKaydedildi: outputs/recovery_curves.png")