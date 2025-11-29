# Feature Engineering

## Genel Bakis

Feature engineering, model performansini artirmak icin yeni degiskenler turetme ve mevcut degiskenleri donusturme surecidir.

## Faz 1: [Faz Adi]

### Turetilen Feature'lar

| Feature Adi | Aciklama | Formul |
|-------------|----------|--------|
| [FEATURE_1] | ... | ... |
| [FEATURE_2] | ... | ... |

### Performans Etkisi

- **Onceki AUC:** X.XXXX
- **Sonraki AUC:** X.XXXX
- **Gelisim:** +X.XX%

## Faz 2: [Faz Adi]

### Turetilen Feature'lar

| Feature Adi | Aciklama | Formul |
|-------------|----------|--------|
| [FEATURE_3] | ... | ... |
| [FEATURE_4] | ... | ... |

### Performans Etkisi

- **Onceki AUC:** X.XXXX
- **Sonraki AUC:** X.XXXX
- **Gelisim:** +X.XX%

## Feature Secimi

### Yontem

1. **Istatistiksel Filtreleme**
   - Dusuk varyansli feature'larin cikarilmasi
   - Yuksek korelasyonlu feature'larin cikarilmasi

2. **Importance-Based Filtreleme**
   - LightGBM feature importance
   - SHAP degerleri

### Sonuc

- **Baslangic Feature Sayisi:** X
- **Final Feature Sayisi:** Y
- **Cikartilan Feature Sayisi:** Z

## En Onemli Feature'lar

| Sira | Feature | Importance |
|------|---------|------------|
| 1 | [FEATURE_1] | X.XXX |
| 2 | [FEATURE_2] | X.XXX |
| 3 | [FEATURE_3] | X.XXX |
| ... | ... | ... |

## Kod

Detayli kod icin: `notebooks/03_feature_engineering.ipynb`

