"""USGS dNBR yanık şiddeti sınıfları (Key & Benson, 2006)."""
import numpy as np

SEVERITY_BOUNDS = [-0.25, -0.1, 0.1, 0.27, 0.44, 0.66]
SEVERITY_LABELS = [
    "Yüksek yeniden büyüme", "Düşük yeniden büyüme", "Yanmamış",
    "Düşük şiddet", "Orta-düşük şiddet", "Orta-yüksek şiddet", "Yüksek şiddet",
]
SEVERITY_COLORS = ["#0b6623", "#66bb6a", "#fff9c4", "#ffd54f", "#ff8a65", "#e53935", "#7b0000"]
BURNED_CLASSES = [3, 4, 5, 6]  # sadece gerçekten yanmış sınıflar
BURNED_LABELS = [SEVERITY_LABELS[i] for i in BURNED_CLASSES]


def classify_severity(dnbr_values):
    """dNBR dizisini USGS sınıflarına ayırır; NaN pikseller -1 olur."""
    dnbr_values = np.asarray(dnbr_values)
    cls = np.digitize(dnbr_values, SEVERITY_BOUNDS)
    return np.where(np.isnan(dnbr_values), -1, cls)
