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
git clone https://github.com/f0rknturkoglu/zero2endMLbootcamp.git
cd zero2endMLbootcamp
```

### 2. Virtual Environment Olusturma

```bash
# venv ile
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# veya conda ile
conda create -n bank-marketing python=3.10
conda activate bank-marketing
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
kaggle datasets download -d janiobachmann/bank-marketing-dataset
unzip bank-marketing-dataset.zip -d data/raw/
```

#### Manuel:

1. [Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) adresinden veri setini indirin
2. `data/raw/` klasorune cikartin (`bank.csv` dosyasi olmali)

### 5. Konfigurasyon

`src/config.py` dosyasi varsayilan olarak Kaggle Bank Marketing datasetine gore ayarlanmistir. Ozel bir ayar yapmaniza gerek yoktur.

## Calistirma

### Notebook'lari Calistirma

```bash
jupyter notebook
# veya
jupyter lab
```

Sırasıyla `notebooks/` klasorundeki notebook'lari calistirin:
1. `01_eda.ipynb`
2. `02_baseline.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_model_optimization.ipynb`
5. `05_model_evaluation.ipynb`
6. `06_final_pipeline.ipynb`

### Pipeline'i Calistirma (Tek Seferde Egitim)

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
# Arayuz: http://localhost:7860
```

## Deployment (Hugging Face Spaces)

Bu proje Hugging Face Spaces uzerinde calismaktadir.

**Canlı Demo:** [https://huggingface.co/spaces/f0rknturkoglu/bank-marketing-prediction](https://huggingface.co/spaces/f0rknturkoglu/bank-marketing-prediction)

### Manuel Deployment Adimlari

1. Hugging Face Space olusturun (SDK: Gradio)
2. Bu repoyu Space'e pushlayin:

```bash
git remote add huggingface https://huggingface.co/spaces/f0rknturkoglu/bank-marketing-prediction
git push huggingface main
```

## Sorun Giderme

### Hata: Model dosyasi bulunamadi
Once notebook'lari veya `src/pipeline.py` scriptini calistirip modeli egitmeniz gerekiyor.

### Hata: Veri dosyasi bulunamadi
`data/raw/` klasorune `bank.csv` dosyasini koydugunuzdan emin olun.

### Hata: Import hatasi
Virtual environment'in aktif oldugunu ve bagimliliklarin yuklendigini kontrol edin.
