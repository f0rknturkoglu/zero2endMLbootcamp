# Model Değerlendirme

## 1. Performans Metrikleri (Production-Ready Model)

### Final Model Sonuçları (Test Seti)

| Metrik | Değer |
|--------|-------|
| **AUC** | 0.7828 |
| **Accuracy** | 72.59% |
| **Precision** | 75.34% |
| **Recall** | 62.67% |
| **F1 Score** | 68.42% |

### Confusion Matrix

| | Predicted 0 (No) | Predicted 1 (Yes) |
|---|---|---|
| **Actual 0 (No)** | 958 (TN) | 217 (FP) |
| **Actual 1 (Yes)** | 395 (FN) | 663 (TP) |

### Cross-Validation Sonuçları (5-Fold)

- **Ortalama AUC:** 0.7845 (+/- 0.0144)
- **Min AUC:** 0.7654
- **Max AUC:** 0.7992

## 2. Feature Importance

### Top 5 Feature (SHAP & Model Importance)

1. **age_campaign_interaction** (1938) - Yaş ve kampanya etkileşimi
2. **day** (1731) - Ayın günü
3. **age_balance_interaction** (1526) - Yaş ve bakiye etkileşimi
4. **balance_per_age** (1502) - Yaş başına düşen bakiye
5. **balance** (1481) - Yıllık ortalama bakiye

### SHAP Analizi Notları
- **Interaction Feature'ları:** Türetilen feature'ların (interaction) en üst sıralarda yer alması, feature engineering başarısını gösterir.
- **Mevsimsellik:** `day` ve `month` değişkenlerinin önemi, kampanyanın zamanlamasının kritik olduğunu doğrular.

## 3. Baseline vs Final Model Karşılaştırması

| Metrik | Baseline (duration ile) | Final (duration olmadan) | Fark | Fark % |
|--------|-------------------------|--------------------------|------|--------|
| **AUC** | 0.9259 | 0.7828 | -0.1431 | -15.45% |
| **Accuracy** | 0.8607 | 0.7259 | -0.1348 | -15.66% |
| **Precision** | 0.8297 | 0.7534 | -0.0763 | -9.19% |
| **Recall** | 0.8885 | 0.6267 | -0.2618 | -29.47% |
| **F1** | 0.8581 | 0.6842 | -0.1739 | -20.26% |

> **⚠️ ÖNEMLİ NOT:** Baseline modelin daha yüksek skor vermesinin sebebi, production ortamında önceden bilinmesi imkansız olan `duration` (görüşme süresi) değişkenini içermesidir. Final model, **gerçekçi bir production senaryosu** için `duration` olmadan eğitilmiştir ve 0.78 AUC ile başarılı kabul edilmektedir.

## 4. Business Uyumluluğu ve Senaryo Analizi

### Eşik Değeri (Threshold) Analizi

| Threshold | Precision | Recall | F1 | Net Kar (Tahmini) | ROI |
|-----------|-----------|--------|----|-------------------|-----|
| 0.3 | 60.92% | 62.67% | 70.09% | €80,135 | 1118% |
| 0.5 | 75.34% | 62.67% | 68.42% | €47,605 | 1589% |

*Not: Optimal Threshold (Youden's J) 0.55 civarındadır. Threshold 0.3 toplam karı maksimize ederken, Threshold 0.5 verimliliği (ROI) maksimize etmektedir.*

### Bank Marketing Senaryosu
- **Maliyet:** Her arama ~5€
- **Gelir:** Başarılı dönüşüm ~100€
- **Strateji:** Eğer bankanın çağrı merkezi kapasitesi genişse **Threshold 0.3** seçilerek toplam kar maksimize edilebilir. Eğer kapasite kısıtlıysa ve verimlilik önemliyse **Threshold 0.5** tercih edilmelidir.

### Müşteri Risk Segmentasyonu

Model tahminlerine göre müşteriler segmentlere ayrılmıştır:

| Segment | Oran | Dönüşüm Oranı (Conversion Rate) |
|---------|------|---------------------------------|
| **Düşük Risk** | %35.83 | %23.12 |
| **Orta Risk** | %24.76 | %37.97 |
| **Yüksek Risk** | %12.58 | %55.87 |
| **Çok Yüksek Risk** | %26.82 | %84.47 |

## 5. Sonuçlar ve Öneriler

1. **Hedefleme:** "Çok Yüksek Risk" ve "Yüksek Risk" segmentlerine öncelik verilmeli. Bu segmentlerde dönüşüm oranı %55-85 arasındadır.
2. **Zamanlama:** Ayın günü (`day`) ve mevsimsellik (`month`) dikkate alınarak aramalar planlanmalı.
3. **Feature Engineering:** Yaş ve bakiye etkileşimleri modelin başarısını artırmıştır.
4. **Deployment:** Model API olarak canlıya alınmalı ve CRM sistemine entegre edilmelidir.
