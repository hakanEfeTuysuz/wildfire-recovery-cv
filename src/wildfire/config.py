"""Proje genelindeki yollar, çalışma alanı ve tarih sabitleri.

Tüm dosya yolları depo kök dizinine göre mutlak olarak tanımlanır; betikler
hangi dizinden çalıştırılırsa çalıştırılsın aynı yerlere okur/yazar.
"""
from pathlib import Path

# --- Dizinler ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"      # büyük ara veriler (git'e eklenmez)
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
LOGS_DIR = OUTPUT_DIR / "logs"

for _d in (INTERIM_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Ara veri dosyaları ---
DNBR_TIF = INTERIM_DIR / "dnbr_manavgat_2021.tif"
RECOVERY_DATASET = INTERIM_DIR / "recovery_dataset.npz"

# --- Çalışma alanı ---
BBOX_4326 = [30.95, 36.65, 31.75, 37.20]   # boylam/enlem, WGS84
UTM_CRS = "EPSG:32636"                      # WGS 84 / UTM 36N

# --- Yangın referans tarihleri ---
PRE_FIRE_DATE = "2021-07-20"
POST_FIRE_DATE = "2021-08-14"               # toparlanma için t = 0
TARGET_YEAR = 2026                          # toparlanma düzeyinin ölçüldüğü yıl

# Önbelleklenen ara çıktıların sürümü. Ön işleme mantığı değiştiğinde artırılır ki
# eski (hatalı) önbellekler sessizce yeniden kullanılmasın.
PIPELINE_VERSION = 2


def nbr_cache_path(date):
    """het_reliability.py'nin indirdiği ek tarihler için NBR önbellek dosyası."""
    return INTERIM_DIR / f"nbr_cache_v{PIPELINE_VERSION}_{date}.npy"
