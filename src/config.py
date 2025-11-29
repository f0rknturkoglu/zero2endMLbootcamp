"""
Proje Konfigurasyonu
====================
Bank Marketing Dataset - Vadeli Mevduat Tahmini
Tum path'ler, business kurallari ve model ayarlari burada tanimlanir.
"""

from pathlib import Path
import os

# ============================================================================
# PATH AYARLARI
# ============================================================================

# Proje root dizini
PROJECT_ROOT = Path(__file__).parent.parent

# Data path'leri
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model path'leri
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_DIR = MODELS_DIR / "final"

# Notebook path'leri
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Docs path'leri
DOCS_DIR = PROJECT_ROOT / "docs"

# ============================================================================
# DATA AYARLARI
# ============================================================================

# Bank Marketing Dataset
# Kaynak: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset
TRAIN_DATA = RAW_DATA_DIR / "bank.csv"

# Target degisken (deposit: musteri vadeli mevduat acti mi?)
# yes/no -> 1/0 olarak encode edilecek
TARGET_COLUMN = "deposit"

# Bu datasette ID sutunu yok
ID_COLUMN = None

# ============================================================================
# FEATURE BILGILERI
# ============================================================================

# Numerik sutunlar
NUMERIC_FEATURES = [
    "age",           # Musteri yasi
    "balance",       # Yillik ortalama bakiye (euro)
    "day",           # Son iletisim gunu (ayin gunu)
    "duration",      # Son iletisim suresi (saniye) - DIKKAT: Production'da cikarilmali!
    "campaign",      # Bu kampanyada yapilan iletisim sayisi
    "pdays",         # Onceki kampanyadan bu yana gecen gun (-1: daha once iletisim yok)
    "previous",      # Onceki kampanyalarda yapilan iletisim sayisi
]

# Kategorik sutunlar
CATEGORICAL_FEATURES = [
    "job",           # Meslek turu
    "marital",       # Medeni durum
    "education",     # Egitim seviyesi
    "default",       # Kredi temerrut durumu
    "housing",       # Konut kredisi var mi?
    "loan",          # Bireysel kredi var mi?
    "contact",       # Iletisim turu
    "month",         # Son iletisim ayi
    "poutcome",      # Onceki kampanya sonucu
]

# ============================================================================
# MODEL AYARLARI
# ============================================================================

# Random state (reproducibility icin)
RANDOM_STATE = 42

# Cross-validation ayarlari
CV_FOLDS = 5
CV_SHUFFLE = True

# Train-test split orani
TEST_SIZE = 0.2

# ============================================================================
# LIGHTGBM DEFAULT PARAMETRELERI
# ============================================================================

LGBM_DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# ============================================================================
# OPTUNA AYARLARI
# ============================================================================

OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 saat

# ============================================================================
# BUSINESS KURALLARI
# ============================================================================

# Vadeli mevduat acma olasiligi esikleri
DEPOSIT_PROBABILITY_THRESHOLDS = {
    "dusuk": 0.3,      # %30 altinda - aramaya degmez
    "orta": 0.5,       # %30-50 arasi - normal oncelik
    "yuksek": 0.7,     # %50-70 arasi - yuksek oncelik
    "cok_yuksek": 0.7  # %70 ustu - oncelikli ara
}

# Kampanya stratejisi
MAX_CAMPAIGN_CALLS = 5  # Bir musteriye maksimum arama sayisi
MIN_DAYS_BETWEEN_CAMPAIGNS = 7  # Kampanyalar arasi minimum gun

# ONEMLI NOT: 'duration' (gorusme suresi) feature'i
# Bu feature tahmin sirasinda BILINMEZ cunku gorusme henuz yapilmadi.
# Production modelinde bu feature CIKARILMALIDIR!
EXCLUDE_IN_PRODUCTION = ["duration"]

# ============================================================================
# API AYARLARI
# ============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True  # Development icin True, production'da False

# ============================================================================
# LOGGING AYARLARI
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

