# API Deployment

## Genel Bakis

Bu dokuman, ML modelinin REST API olarak deploy edilmesi surecini aciklar.

## API Mimarisi

```
Client --> FastAPI --> Inference Module --> Model
                            |
                            v
                      Preprocessor
```

## Endpoint'ler

### GET /

Ana sayfa ve API bilgisi.

**Response:**
```json
{
    "message": "ML Model API",
    "docs": "/docs",
    "health": "/health"
}
```

### GET /health

Saglik kontrolu.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### POST /predict

Tek bir ornek icin tahmin.

**Request:**
```json
{
    "features": {
        "feature1": 1.0,
        "feature2": 2.0,
        ...
    }
}
```

**Response:**
```json
{
    "prediction": 0,
    "probability": 0.23,
    "risk_level": "Dusuk Risk"
}
```

### POST /predict/batch

Toplu tahmin.

**Request:**
```json
{
    "data": [
        {"feature1": 1.0, "feature2": 2.0},
        {"feature1": 3.0, "feature2": 4.0}
    ]
}
```

**Response:**
```json
{
    "predictions": [
        {"prediction": 0, "probability": 0.23, "risk_level": "Dusuk Risk"},
        {"prediction": 1, "probability": 0.78, "risk_level": "Yuksek Risk"}
    ]
}
```

## Local Calistirma

```bash
# API'yi baslat
python src/app.py

# Swagger UI
http://localhost:8000/docs
```

## Deployment Secenekleri

### 1. Render

1. Render hesabi olusturun
2. Yeni Web Service olusturun
3. GitHub repo'nuzu baglayin
4. Build ve Start komutlarini ayarlayin:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`

### 2. Hugging Face Spaces

1. Hugging Face hesabi olusturun
2. Yeni Space olusturun (Gradio veya Docker)
3. Dosyalari yukleyin

### 3. Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring

### Izlenmesi Gereken Metrikler

1. **Performans Metrikleri**
   - Response time (latency)
   - Throughput (requests/second)
   - Error rate

2. **Model Metrikleri**
   - Tahmin dagilimi
   - Feature dagilimi (data drift)
   - Model performansi (eger ground truth varsa)

3. **Sistem Metrikleri**
   - CPU/Memory kullanimi
   - Disk kullanimi

## Guvenlik

- API key authentication (production icin)
- Rate limiting
- Input validation
- HTTPS

