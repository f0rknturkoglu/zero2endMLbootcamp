# Model Optimizasyonu

## Hiperparametre Optimizasyonu

### Kullanilan Yontem

- **Arac:** Optuna
- **Deneme Sayisi:** X
- **Timeout:** X saat

### Optimize Edilen Parametreler

| Parametre | Aralik | Optimal Deger |
|-----------|--------|---------------|
| n_estimators | [100, 2000] | X |
| learning_rate | [0.01, 0.3] | X.XX |
| num_leaves | [20, 100] | X |
| max_depth | [3, 15] | X |
| min_child_samples | [10, 100] | X |
| subsample | [0.6, 1.0] | X.XX |
| colsample_bytree | [0.6, 1.0] | X.XX |
| reg_alpha | [0, 10] | X.XX |
| reg_lambda | [0, 10] | X.XX |

### Optimizasyon Sureci

[Optuna optimizasyon grafigi ve aciklamasi]

## Model Karsilastirmasi

| Model | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| Logistic Regression | X.XXXX | X.XX% | X.XX% | X.XX% |
| Random Forest | X.XXXX | X.XX% | X.XX% | X.XX% |
| LightGBM | X.XXXX | X.XX% | X.XX% | X.XX% |
| XGBoost | X.XXXX | X.XX% | X.XX% | X.XX% |
| CatBoost | X.XXXX | X.XX% | X.XX% | X.XX% |

## Final Model

- **Secilen Model:** [LightGBM]
- **Secim Nedeni:** [En yuksek AUC, hizli egitim, vb.]

### Final Parametreler

```python
{
    "objective": "binary",
    "metric": "auc",
    "n_estimators": X,
    "learning_rate": X.XX,
    "num_leaves": X,
    # ...
}
```

## Kod

Detayli kod icin: `notebooks/04_model_optimization.ipynb`

