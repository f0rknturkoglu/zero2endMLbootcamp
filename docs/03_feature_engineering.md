# Feature Engineering

## 5. Sonuçlar ve Özet

### Performans Karşılaştırması

| Aşama | Feature Sayısı | CV AUC | Açıklama |
|-------|----------------|--------|----------|
| **Baseline (duration ile)** | 16 | 0.9239 | Gerçekçi değil (duration production'da bilemeyiz.) |
| **Faz 1** | 18 | 0.7907 | duration çıkarıldı, temel feature'lar eklendi |
| **Faz 2** | 30 | 0.7897 | Tüm yeni feature'lar eklendi |
| **Final (Seçilmiş)** | 20 | 0.7880 | Feature selection sonrası |

### Önemli Bulgular

1. **duration Feature'inin Etkisi:**
   - Baseline'da en önemli feature'lardan biriydi
   - Production'da kullanılamaz (görüşme yapılmadan önce bilinmez)
   - Çıkarılınca performans düştü: 0.9239 → 0.7907 (-14.4%)
   - bu beklediğimiz bir şeydi.

2. **Feature Engineering Etkisi:**
   - Faz 1'de yas grupları, bakiye kategorileri eklendi
   - Faz 2'de mevsimsellik, interaksiyon, ratio feature'ları eklendi
   - Toplam 14 yeni feature türetildi

3. **Feature Selection:**
   - 30 feature'dan 20 feature'a düşürüldü
   - Düşük varyanslı ve yüksek korelasyonlu feature'lar çıkarıldı
   - En önemli 20 feature seçildi

### En Önemli Feature'lar (SON)
1. **month** (356) - Mevsimsellik
2. **day** (355) - Ayın günü
3. **age** (291) - Müşteri yaşı
4. **age_campaign_interaction** (287) - Yaş × Kampanya
5. **balance_per_age** (252) - Yaş başına bakiye

### Sonraki Adımlar

1. **Model Optimizasyonu** (04_model_optimization.ipynb)
   - Hiperparametre optimizasyonu (Optuna)
   - Model karşılaştırması
   - Final model seçimi

2. **Model Değerlendirme** (05_model_evaluation.ipynb)
   - SHAP analizi
   - Business uyumluluk kontrolü
   - Threshold optimizasyonu

### Not

**Gerçekçi bir production modeli için:**
- `duration` feature'i kullanılamaz
- Final AUC: 0.7880 (duration olmadan)
- Bu skor, gerçek dünya senaryosunda kullanılabilir bir modeldir
- Baseline'dan fark: -14.71% (duration'ın etkisi büyük ama gerçekçi değil)
