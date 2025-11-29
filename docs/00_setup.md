# Kurulum ve Baslangic

Bu dokuman projenin kurulumu ve calistirilmasi icin gerekli adimlari icerir.

## Gereksinimler

- Python 3.10+
- pip veya conda
- Git
- (Opsiyonel) Kaggle API

## Kurulum Adimlari

### 1. Repository'yi Klonlama

```bash
git clone https://github.com/[KULLANICI_ADI]/[REPO_ADI].git
cd [REPO_ADI]
```

### 2. Virtual Environment Olusturma

```bash
# venv ile
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# veya conda ile
conda create -n ml-project python=3.10
conda activate ml-project
```

### 3. Bagimliliklari Yukleme

```bash
pip install -r requirements.txt
```

### 4. Veri Setini Indirme

#### Kaggle API ile:

1. Kaggle hesabinizdan API token indirin (`kaggle.json`)
2. Token'i `~/.kaggle/` klasorune koyun
3. Komutu calistirin:

```bash
kaggle competitions download -c [YARISMA_ADI]
unzip [YARISMA_ADI].zip -d data/raw/
```

#### Manuel:

1. Kaggle'dan veri setini indirin
2. `data/raw/` klasorune cikartin

### 5. Konfigurasyon

`src/config.py` dosyasini acin ve gerekli path'leri guncelleyin:

```python
TRAIN_DATA = RAW_DATA_DIR / "application_train.csv"  # Kendi dosya adiniz
TARGET_COLUMN = "TARGET"  # Kendi target sutununuz
ID_COLUMN = "SK_ID_CURR"  # Kendi ID sutununuz
```

## Calistirma

### Notebook'lari Calistirma

```bash
jupyter notebook
# veya
jupyter lab
```

### Pipeline'i Calistirma

```bash
python src/pipeline.py
```

### API'yi Calistirma

```bash
python src/app.py
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Gradio Arayuzunu Calistirma

```bash
python src/app.py gradio
```

## Sorun Giderme

### Hata: Model dosyasi bulunamadi

Once notebook'lari calistirip modeli egitmeniz gerekiyor.

### Hata: Veri dosyasi bulunamadi

`data/raw/` klasorune veri dosyalarinizi koydugunuzdan emin olun.

### Hata: Import hatasi

Virtual environment'in aktif oldugunu ve bagimliliklarin yuklendigini kontrol edin.

