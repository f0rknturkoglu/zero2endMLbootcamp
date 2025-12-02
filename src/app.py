"""
Web Application
===============
REST API (FastAPI) ve arayuz (Gradio).

Bank Marketing - Vadeli Mevduat Tahmini API ve Demo Arayuzu
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import uvicorn
import gradio as gr
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

from config import API_HOST, API_PORT, FINAL_MODEL_DIR
from inference import ModelInference

# Logging ayarlari
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI UYGULAMASI
# ============================================================================

app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Bank Marketing Dataset - Vadeli Mevduat Tahmini API. Musterilerin vadeli mevduat acip acmayacagini tahmin eden makine ogrenmesi modeli.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS ayarlari
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model inference objesi
inference = ModelInference()


class PredictionRequest(BaseModel):
    """Tahmin istegi modeli."""
    features: Dict[str, Any] = Field(
        ...,
        description="Musteri bilgilerini iceren feature dictionary. Gerekli alanlar: age, job, marital, education, default, balance, housing, loan, contact, day, month, campaign, pdays, previous, poutcome"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Feature'larin gecerli oldugunu kontrol et."""
        required_fields = [
            'age', 'job', 'marital', 'education', 'default', 
            'balance', 'housing', 'loan', 'contact', 'day', 
            'month', 'campaign', 'pdays', 'previous', 'poutcome'
        ]
        missing = [field for field in required_fields if field not in v]
        if missing:
            raise ValueError(f"Eksik feature'lar: {', '.join(missing)}")
        return v


class PredictionResponse(BaseModel):
    """Tahmin yaniti modeli."""
    prediction: int
    probability: float
    risk_level: str


class BatchPredictionRequest(BaseModel):
    """Toplu tahmin istegi modeli."""
    data: List[Dict[str, Any]]


class BatchPredictionResponse(BaseModel):
    """Toplu tahmin yaniti modeli."""
    predictions: List[PredictionResponse]


@app.on_event("startup")
async def startup_event():
    """Uygulama basladiginda modeli yukle."""
    try:
        inference.load_model()
        logger.info("Model basariyla yuklendi")
    except FileNotFoundError as e:
        logger.warning(f"Model dosyasi bulunamadi: {e}. /predict endpoint'i calismayacak.")
    except Exception as e:
        logger.error(f"Model yuklenirken hata: {e}")


@app.get("/")
async def root():
    """Ana sayfa."""
    return {
        "message": "Bank Marketing Prediction API",
        "description": "Vadeli mevduat tahmini icin makine ogrenmesi modeli API'si",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/model/info"
        }
    }


@app.get("/health")
async def health_check():
    """Saglik kontrolu."""
    model_status = inference.model is not None
    return {
        "status": "healthy" if model_status else "degraded",
        "model_loaded": model_status,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Tek bir ornek icin tahmin yap.
    
    Musteri bilgilerine gore vadeli mevduat acma olasiligini tahmin eder.
    
    Args:
        request: Feature dictionary iceren istek
        
    Returns:
        Tahmin sonucu (prediction, probability, risk_level)
        
    Raises:
        HTTPException: Model yuklenmemisse veya gecersiz input varsa
    """
    if inference.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model yuklenmedi. Lutfen modeli egitip kaydedin."
        )
    
    try:
        logger.info(f"Tahmin istegi alindi: {request.features.get('age', 'N/A')} yasinda musteri")
        result = inference.predict_single(request.features)
        logger.info(f"Tahmin tamamlandi: prediction={result['prediction']}, probability={result['probability']:.4f}")
        return PredictionResponse(**result)
    except ValueError as e:
        logger.error(f"Gecersiz input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gecersiz input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Tahmin sirasinda hata: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin sirasinda bir hata olustu: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Birden fazla ornek icin toplu tahmin yap.
    
    Args:
        request: Feature dictionary listesi iceren istek (maksimum 1000 ornek)
        
    Returns:
        Tahmin sonuclari listesi
        
    Raises:
        HTTPException: Model yuklenmemisse, gecersiz input varsa veya cok fazla ornek varsa
    """
    if inference.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model yuklenmedi. Lutfen modeli egitip kaydedin."
        )
    
    if len(request.data) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maksimum 1000 ornek tahmin edilebilir."
        )
    
    try:
        logger.info(f"Toplu tahmin istegi alindi: {len(request.data)} ornek")
        predictions = []
        errors = []
        
        for idx, features in enumerate(request.data):
            try:
                result = inference.predict_single(features)
                predictions.append(PredictionResponse(**result))
            except Exception as e:
                errors.append(f"Ornek {idx}: {str(e)}")
                logger.warning(f"Ornek {idx} tahmin edilemedi: {e}")
        
        if errors and not predictions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tum orneklerde hata: {'; '.join(errors)}"
            )
        
        logger.info(f"Toplu tahmin tamamlandi: {len(predictions)}/{len(request.data)} basarili")
        return BatchPredictionResponse(predictions=predictions)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Toplu tahmin sirasinda hata: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Toplu tahmin sirasinda bir hata olustu: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """
    Model bilgilerini getir.
    
    Returns:
        Model tipi, feature isimleri ve diger metadata
    """
    if inference.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model yuklenmedi"
        )
    
    return {
        "model_type": type(inference.model).__name__,
        "feature_count": len(inference.feature_names) if inference.feature_names else 0,
        "feature_names": inference.feature_names,
        "label_encoder_count": len(inference.label_encoders) if inference.label_encoders else 0
    }


# ============================================================================
# GRADIO ARAYUZU
# ============================================================================

# Custom CSS
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 20px;
    color: white;
}
.result-positive {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    font-size: 1.2em;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    font-size: 1.2em;
    text-align: center;
}
.info-box {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    text-align: center;
}
"""

def create_gradio_interface():
    """Gradio arayuzu olustur."""
    
    # Monitoring DB
    from src.monitoring import MonitoringDB
    monitor = MonitoringDB()

    def predict_gradio(age, job, marital, education, default, balance, 
                      housing, loan, contact, day, month, campaign, 
                      pdays, previous, poutcome):
        """Gradio tahmin fonksiyonu."""
        if inference.model is None:
            return "❌ Model yüklenmedi!", "", "⚠️ Bilinmiyor", ""
        
        # Features dictionary olustur
        features = {
            "age": int(age) if age is not None else 45,
            "job": str(job) if job else "management",
            "marital": str(marital) if marital else "married",
            "education": str(education) if education else "tertiary",
            "default": str(default) if default else "no",
            "balance": float(balance) if balance is not None else 1500.0,
            "housing": str(housing) if housing else "yes",
            "loan": str(loan) if loan else "no",
            "contact": str(contact) if contact else "cellular",
            "day": int(day) if day is not None else 15,
            "month": str(month) if month else "may",
            "campaign": int(campaign) if campaign is not None else 2,
            "pdays": int(pdays) if pdays is not None else -1,
            "previous": int(previous) if previous is not None else 0,
            "poutcome": str(poutcome) if poutcome else "unknown"
        }
        
        try:
            result = inference.predict_single(features)
            
            # Loglama
            monitor.log_prediction(features, result)
            
            prob = result['probability']
            
            # Emoji ve renk bazlı sonuç
            if result['prediction'] == 1:
                prediction_text = "✅ VADELİ MEVDUAT AÇABİLİR"
                action = "📞 Bu müşteri aranmalı!"
            else:
                prediction_text = "❌ VADELİ MEVDUAT AÇMAYACAK"
                action = "⏸️ Bu müşteri öncelikli değil"
            
            # Olasılık gösterimi
            prob_bar = "🟩" * int(prob * 10) + "⬜" * (10 - int(prob * 10))
            probability_text = f"{prob_bar} {prob:.1%}"
            
            # Risk seviyesi emoji
            risk_map = {
                "Dusuk Risk": "🟢 Düşük Risk",
                "Orta Risk": "🟡 Orta Risk", 
                "Yuksek Risk": "🟠 Yüksek Risk",
                "Cok Yuksek Risk": "🔴 Çok Yüksek Risk"
            }
            risk_text = risk_map.get(result['risk_level'], result['risk_level'])
            
            # Detaylı açıklama
            explanation = f"""
## 📊 Analiz Sonucu

| Metrik | Değer |
|--------|-------|
| **Tahmin** | {prediction_text} |
| **Olasılık** | {prob:.2%} |
| **Risk Seviyesi** | {risk_text} |
| **Önerilen Aksiyon** | {action} |

---

### 💡 Yorumlama Rehberi

- **%70+**: Çok yüksek potansiyel - Öncelikli aranmalı
- **%50-70**: Yüksek potansiyel - Aranmalı  
- **%30-50**: Orta potansiyel - Normal sırada
- **%30 altı**: Düşük potansiyel - Önceliksiz
"""
            
            return (
                prediction_text,
                probability_text,
                risk_text,
                explanation
            )
        except Exception as e:
            return f"❌ Hata: {str(e)}", "", "⚠️ Hata", ""

    def get_monitoring_stats():
        """Monitoring istatistiklerini getir"""
        stats = monitor.get_stats()
        recent_logs = monitor.get_recent_logs()
        
        stats_html = f"""
        <div style="display: flex; gap: 20px; margin-bottom: 20px;">
            <div class="metric-card" style="flex: 1;">
                <div style="font-size: 2em; color: #007bff;">{stats['total_predictions']}</div>
                <div style="color: #666;">Toplam Tahmin</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div style="font-size: 2em; color: #28a745;">%{stats['positive_rate']:.1f}</div>
                <div style="color: #666;">Pozitif Oranı</div>
            </div>
            <div class="metric-card" style="flex: 1;">
                <div style="font-size: 2em; color: #ffc107;">%{stats['avg_probability']*100:.1f}</div>
                <div style="color: #666;">Ortalama Olasılık</div>
            </div>
        </div>
        """
        
        return stats_html, recent_logs

    # Input'lar - Bank Marketing Dataset feature'ları
    with gr.Blocks(css=CUSTOM_CSS) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🏦 Bank Marketing - Vadeli Mevduat Tahmini</h1>
            <p style="font-size: 1.1em; opacity: 0.9;">
                Yapay zeka destekli müşteri analizi ile vadeli mevduat açma olasılığını tahmin edin
            </p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.TabItem("🎯 Tahmin"):
                with gr.Row():
                    # Sol Panel - Müşteri Bilgileri
                    with gr.Column(scale=1):
                        gr.HTML('<h3>👤 Müşteri Demografik Bilgileri</h3>')
                        
                        with gr.Group():
                            age = gr.Slider(label="Yaş", minimum=18, maximum=95, value=45, step=1)
                            job = gr.Dropdown(label="Meslek", choices=["management", "technician", "entrepreneur", "blue-collar", "unknown", "retired", "admin", "services", "self-employed", "unemployed", "housemaid", "student"], value="management")
                            marital = gr.Radio(label="Medeni Durum", choices=["married", "single", "divorced"], value="married")
                            education = gr.Dropdown(label="Eğitim Seviyesi", choices=["tertiary", "secondary", "primary", "unknown"], value="tertiary")
                        
                        gr.HTML('<h3>💰 Finansal Durum</h3>')
                        
                        with gr.Group():
                            balance = gr.Number(label="Yıllık Ortalama Bakiye (€)", value=1500)
                            default = gr.Radio(label="Kredi Temerrüt Geçmişi", choices=["no", "yes"], value="no")
                            housing = gr.Radio(label="Konut Kredisi Var mı?", choices=["yes", "no"], value="yes")
                            loan = gr.Radio(label="Bireysel Kredi Var mı?", choices=["yes", "no"], value="no")
                    
                    # Orta Panel - Kampanya Bilgileri
                    with gr.Column(scale=1):
                        gr.HTML('<h3>📞 Kampanya Bilgileri</h3>')
                        
                        with gr.Group():
                            contact = gr.Dropdown(label="İletişim Türü", choices=["cellular", "telephone", "unknown"], value="cellular")
                            with gr.Row():
                                day = gr.Slider(label="İletişim Günü", minimum=1, maximum=31, value=15, step=1)
                                month = gr.Dropdown(label="İletişim Ayı", choices=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], value="may")
                            campaign = gr.Slider(label="Bu Kampanyada Yapılan Arama Sayısı", minimum=1, maximum=50, value=2, step=1)
                        
                        gr.HTML('<h3>📜 Önceki Kampanya Geçmişi</h3>')
                        
                        with gr.Group():
                            pdays = gr.Number(label="Önceki Kampanyadan Bu Yana Geçen Gün", value=-1)
                            previous = gr.Slider(label="Önceki Kampanyalardaki Toplam Arama", minimum=0, maximum=50, value=0, step=1)
                            poutcome = gr.Dropdown(label="Önceki Kampanya Sonucu", choices=["success", "failure", "other", "unknown"], value="unknown")
                        
                        # Tahmin Butonu
                        gr.HTML('<br>')
                        predict_btn = gr.Button("🎯 Tahmin Yap", variant="primary", size="lg")
                    
                    # Sağ Panel - Sonuçlar
                    with gr.Column(scale=1):
                        gr.HTML('<h3>📊 Tahmin Sonucu</h3>')
                        
                        with gr.Group():
                            prediction = gr.Textbox(label="🎯 Tahmin", interactive=False)
                            probability = gr.Textbox(label="📈 Olasılık", interactive=False)
                            risk_level = gr.Textbox(label="⚠️ Risk Seviyesi", interactive=False)
                        
                        explanation = gr.Markdown(label="📋 Detaylı Analiz")
                
                # Event binding
                predict_btn.click(
                    fn=predict_gradio,
                    inputs=[age, job, marital, education, default, balance,
                           housing, loan, contact, day, month, campaign,
                           pdays, previous, poutcome],
                    outputs=[prediction, probability, risk_level, explanation]
                )

            with gr.TabItem("📈 Monitoring"):
                gr.Markdown("### 📊 Gerçek Zamanlı Model İzleme")
                refresh_btn = gr.Button("🔄 Yenile")
                stats_output = gr.HTML()
                logs_output = gr.Dataframe(
                    headers=["id", "timestamp", "age", "job", "balance", "campaign", "prediction", "probability", "risk_level"],
                    label="Son Tahminler"
                )
                
                refresh_btn.click(
                    fn=get_monitoring_stats,
                    inputs=[],
                    outputs=[stats_output, logs_output]
                )
                
                # İlk yüklemede çalıştır
                interface.load(get_monitoring_stats, None, [stats_output, logs_output])
        
        # Footer
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h3>📈 Model Performans Metrikleri</h3>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 15px; margin-top: 15px;">
                <div class="metric-card" style="flex: 1; min-width: 120px;">
                    <div style="font-size: 2em; color: #007bff;">0.78</div>
                    <div style="color: #666;">AUC-ROC</div>
                </div>
                <div class="metric-card" style="flex: 1; min-width: 120px;">
                    <div style="font-size: 2em; color: #28a745;">0.68</div>
                    <div style="color: #666;">F1 Score</div>
                </div>
                <div class="metric-card" style="flex: 1; min-width: 120px;">
                    <div style="font-size: 2em; color: #ffc107;">0.75</div>
                    <div style="color: #666;">Precision</div>
                </div>
                <div class="metric-card" style="flex: 1; min-width: 120px;">
                    <div style="font-size: 2em; color: #dc3545;">0.63</div>
                    <div style="color: #666;">Recall</div>
                </div>
            </div>
            <p style="text-align: center; margin-top: 15px; color: #666;">
                🤖 Model: LightGBM | 📊 26 Feature | ⚡ Production-Ready (duration feature yok)
            </p>
        </div>
        """)
    
    return interface


# ============================================================================
# MAIN
# ============================================================================

def run_api():
    """FastAPI uygulamasini calistir."""
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )


def run_gradio():
    """Gradio arayuzunu calistir."""
    try:
        inference.load_model()
        logger.info("Model Gradio icin yuklendi")
    except FileNotFoundError as e:
        logger.warning(f"Model dosyasi bulunamadi: {e}")
        print("UYARI: Model dosyasi bulunamadi. Gradio arayuzu calisacak ancak tahmin yapilamayacak.")
    except Exception as e:
        logger.error(f"Model yuklenirken hata: {e}")
    
    interface = create_gradio_interface()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        run_gradio()
    else:
        run_api()

