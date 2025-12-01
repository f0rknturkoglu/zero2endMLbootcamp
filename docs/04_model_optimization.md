# Model Optimizasyonu

## Özet ve Dokümantasyon

### Bu Aşamada Neler Yapıldı?

Bu çalışmada iki farklı gradient boosting algoritmasını (LightGBM ve XGBoost) karşılaştırdık ve hiperparametre optimizasyonu yaptık.

**Adımlar:**
1. Önceki notebook'tan gelen feature set kullanıldı.
2. **Duration** sütunu çıkarıldı çünkü production ortamında bu bilgi mevcut değil.
3. **RandomizedSearchCV** ile 50 farklı parametre kombinasyonu denendi.
4. Regularization parametreleri (L1, L2) eklendi.
5. Her iki model test seti üzerinde değerlendirildi.

### Sonuçlar ve Yorumlar

Duration olmadan elde ettiğimiz **~0.79-0.80 AUC** skoru bu veri seti için gerçekçi ve iyi bir sonuçtur. Duration feature'ı çıkarıldığında performansın düşmesi beklenen bir durumdu çünkü bu değişken target ile çok güçlü bir korelasyona sahipti.

Hiperparametre optimizasyonu baseline'dan belirgin bir iyileşme sağlamadı. Bu da modelin zaten iyi çalıştığını ve veri setinin sınırlarında olduğumuzun bir göstergesi.

### Önemli Kararlar

- **Duration Çıkarıldı:** Gerçek dünya senaryosunda, bir müşteri aramayı cevaplamadan önce görüşme süresini bilemeyiz. Bu yüzden bu feature'ı modelden çıkarmak doğru bir karardı.

- **Regularization:** Overfitting önlemek için L1 ve L2 regularization parametreleri eklendi. Ancak mevcut durumda büyük bir fark yaratmadı.

- **Model Seçimi:** LightGBM ve XGBoost birbirine çok yakın performans gösterdi. Her iki model de production için uygun.
