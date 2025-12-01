# Baseline Model

## 5. Baseline Model Sonuçları

### Cross-Validation Sonuçları
- **5-Fold CV AUC:** 0.9239 (+/- 0.0056)
- Tüm fold'larda tutarlı performans

### Test Seti Sonuçları
- **AUC:** 0.9259
- **Accuracy:** 0.8607 (%86.07)
- **Precision:** 0.8297
- **Recall:** 0.8885
- **F1 Score:** 0.8581

### En Önemli Feature'lar
1. **month** (501) - Mevsimsellik etkisi
2. **duration** (491) - ⚠️ **Production'da kullanılamaz!**
3. **day** (419) - Ayın günü
4. **balance** (339) - Müşteri bakiyesi
5. **age** (296) - Müşteri yaşı

### Önemli Notlar
- Baseline model zaten çok iyi performans gösteriyor (AUC ~0.92)
- `duration` feature'i en önemli feature'lardan biri ama **gerçekçi bir production modeli için çıkarılmalı**
- Feature engineering ile daha da iyileştirilebilir

### Sonraki Adımlar
1. **Feature Engineering** (03_feature_engineering.ipynb)
   - Yas grupları
   - Bakiye kategorileri
   - Mevsimsellik feature'ları
   - `duration` feature'ini çıkararak yeni baseline

2. **Model Optimizasyonu** (04_model_optimization.ipynb)
   - Hiperparametre optimizasyonu
   - Model karşılaştırması
