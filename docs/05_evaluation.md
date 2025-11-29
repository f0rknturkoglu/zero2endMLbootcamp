# Model Degerlendirme

## Performans Metrikleri

### Final Model Sonuclari

| Metrik | Deger |
|--------|-------|
| AUC | X.XXXX |
| Accuracy | X.XX% |
| Precision | X.XX% |
| Recall | X.XX% |
| F1 Score | X.XX% |

### Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| **Actual 0** | TN | FP |
| **Actual 1** | FN | TP |

## Baseline vs Final Model

| Metrik | Baseline | Final | Gelisim |
|--------|----------|-------|---------|
| AUC | X.XXXX | X.XXXX | +X.XX% |
| Accuracy | X.XX% | X.XX% | +X.XX% |
| Precision | X.XX% | X.XX% | +X.XX% |
| Recall | X.XX% | X.XX% | +X.XX% |

## Feature Importance

### Top 10 Feature

| Sira | Feature | Importance | SHAP Value |
|------|---------|------------|------------|
| 1 | [FEATURE_1] | X.XXX | X.XXX |
| 2 | [FEATURE_2] | X.XXX | X.XXX |
| ... | ... | ... | ... |

### SHAP Analizi

[SHAP summary plot ve aciklamasi]

## Business Uyumlulugu

### Esik Degeri Analizi

| Esik | Precision | Recall | F1 |
|------|-----------|--------|-----|
| 0.3 | X.XX% | X.XX% | X.XX% |
| 0.5 | X.XX% | X.XX% | X.XX% |
| 0.7 | X.XX% | X.XX% | X.XX% |

### Business Gereksinimleri

1. **Gereksinim 1:** [Aciklama ve uyumluluk durumu]
2. **Gereksinim 2:** [Aciklama ve uyumluluk durumu]
3. **Gereksinim 3:** [Aciklama ve uyumluluk durumu]

## Model Yorumlanabilirligi

[Modelin yorumlanabilirligi hakkinda degerlendirme]

## Kod

Detayli kod icin: `notebooks/05_model_evaluation.ipynb`

