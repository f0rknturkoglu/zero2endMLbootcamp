# Baseline Model

## Amac

Baseline model, daha karmasik modellerin performansini karsilastirmak icin bir referans noktasi olusturur.

## Kullanilan Yaklasim

### Feature Seti

- Sadece ana tablodaki feature'lar
- Minimal on isleme
- Eksik degerler median/mode ile dolduruldu

### Model

- **Model:** [Logistic Regression / LightGBM / ...]
- **Parametreler:** Default parametreler

### Validasyon

- **Yontem:** [Stratified K-Fold / Train-Test Split]
- **Fold Sayisi:** [5]

## Sonuclar

### Metrikler

| Metrik | Deger |
|--------|-------|
| AUC | X.XXXX |
| Accuracy | X.XX% |
| Precision | X.XX% |
| Recall | X.XX% |
| F1 Score | X.XX% |

### Cross-Validation Sonuclari

| Fold | AUC |
|------|-----|
| 1 | X.XXXX |
| 2 | X.XXXX |
| 3 | X.XXXX |
| 4 | X.XXXX |
| 5 | X.XXXX |
| **Ortalama** | **X.XXXX (+/- X.XXXX)** |

## Cikarimlar

1. [Baseline performansi hakkinda yorum]
2. [Iyilestirme potansiyeli]
3. [Sonraki adimlar]

## Kod

Detayli kod icin: `notebooks/02_baseline.ipynb`

