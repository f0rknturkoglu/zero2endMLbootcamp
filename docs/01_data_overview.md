# Veri Inceleme (Data Overview)

## Veri Seti Hakkinda

- **Kaynak:** Kaggle - Bank Marketing Dataset
- **Orijinal Kaynak:** UCI Machine Learning Repository
- **Link:** https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

Bu veri seti, bir Portekiz bankasinin dogrudan pazarlama kampanyalarindan elde edilmistir. Pazarlama kampanyalari telefon aramalarina dayanmaktadir. Kampanyanin amaci, musterilerin vadeli mevduat hesabi acmasini saglamaktir.

## Veri Seti Yapisi

### Ana Tablo: bank.csv

| Sutun | Tip | Aciklama |
|-------|-----|----------|
| age | int | Musteri yasi |
| job | object | Meslek turu (admin, blue-collar, entrepreneur, vb.) |
| marital | object | Medeni durum (married, single, divorced) |
| education | object | Egitim seviyesi (primary, secondary, tertiary, unknown) |
| default | object | Kredi temerrut durumu (yes, no) |
| balance | int | Yillik ortalama bakiye (euro) |
| housing | object | Konut kredisi var mi? (yes, no) |
| loan | object | Bireysel kredi var mi? (yes, no) |
| contact | object | Iletisim turu (cellular, telephone, unknown) |
| day | int | Son iletisim gunu (ayin kacinci gunu) |
| month | object | Son iletisim ayi |
| duration | int | Son gorusme suresi (saniye) |
| campaign | int | Bu kampanyada yapilan iletisim sayisi |
| pdays | int | Onceki kampanyadan bu yana gecen gun sayisi (-1: daha once iletisim yok) |
| previous | int | Onceki kampanyalarda yapilan iletisim sayisi |
| poutcome | object | Onceki kampanya sonucu (success, failure, other, unknown) |
| deposit | object | **TARGET** - Vadeli mevduat acti mi? (yes, no) |

## Temel Istatistikler

- **Satir Sayisi:** ~45.000
- **Sutun Sayisi:** 17
- **Target Dagilimi:** [EDA'da doldurulacak]

## Kategorik Degisken Kardinaliteleri

| Degisken | Benzersiz Deger Sayisi | Degerler |
|----------|------------------------|----------|
| job | 12 | admin, blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown |
| marital | 3 | married, single, divorced |
| education | 4 | primary, secondary, tertiary, unknown |
| default | 2 | yes, no |
| housing | 2 | yes, no |
| loan | 2 | yes, no |
| contact | 3 | cellular, telephone, unknown |
| month | 12 | jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec |
| poutcome | 4 | success, failure, other, unknown |

## Onemli Notlar

### Duration Feature Hakkinda

`duration` feature'i modelin performansini onemli olcude etkiler cunku uzun sureli gorusmeler genellikle basarili sonuclanir. Ancak:

- Gorusme yapilmadan once bu bilgi **bilinmez**
- Production ortaminda bu feature **kullanilamaz**
- Benchmark amacli kullanilabilir ama gercekci bir model icin cikarilmalidir

### pdays Feature Hakkinda

- Deger -1 ise: Musteri daha once hic aranmamis
- Bu degeri ozel olarak handle etmek gerekebilir (ayri bir flag feature olusturma)

## Onemli Bulgular

[EDA notebook'undan sonra doldurulacak]

1. [Bulgu 1]
2. [Bulgu 2]
3. [Bulgu 3]
