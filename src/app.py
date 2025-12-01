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

def create_gradio_interface():
    """Gradio arayuzu olustur."""
    
    def predict_gradio(age, job, marital, education, default, balance, 
                      housing, loan, contact, day, month, campaign, 
                      pdays, previous, poutcome):
        """Gradio tahmin fonksiyonu."""
        if inference.model is None:
            return "Model yuklenmedi!", 0.0, "Bilinmiyor", ""
        
        # Features dictionary olustur
        features = {
            "age": int(age),
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": float(balance),
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": int(day),
            "month": month,
            "campaign": int(campaign),
            "pdays": int(pdays),
            "previous": int(previous),
            "poutcome": poutcome
        }
        
        try:
            result = inference.predict_single(features)
            
            # Detaylı açıklama
            action = "Arama yapılmalı" if result['prediction'] == 1 else "Arama yapılmamalı"
            explanation = f"""
**Tahmin Detayları:**
- Müşteri vadeli mevduat açma olasılığı: {result['probability']:.2%}
- Risk Seviyesi: {result['risk_level']}
- Önerilen Aksiyon: {action}
"""
            
            prediction_text = "✅ VADELİ MEVDUAT AÇABİLİR" if result['prediction'] == 1 else "❌ VADELİ MEVDUAT AÇMAYABİLİR"
            
            return (
                prediction_text,
                result['probability'],
                result['risk_level'],
                explanation
            )
        except Exception as e:
            return f"Hata: {str(e)}", 0.0, "Hata", ""
    
    # Input'lar - Bank Marketing Dataset feature'ları
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🏦 Bank Marketing - Vadeli Mevduat Tahmini")
        gr.Markdown("Müşterinin vadeli mevduat açıp açmayacağını tahmin edin.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Müşteri Bilgileri")
                age = gr.Number(label="Yaş", value=45, minimum=18, maximum=100)
                job = gr.Dropdown(
                    label="Meslek",
                    choices=["admin", "blue-collar", "entrepreneur", "housemaid", 
                            "management", "retired", "self-employed", "services", 
                            "student", "technician", "unemployed", "unknown"],
                    value="management"
                )
                marital = gr.Dropdown(
                    label="Medeni Durum",
                    choices=["married", "single", "divorced"],
                    value="married"
                )
                education = gr.Dropdown(
                    label="Eğitim Seviyesi",
                    choices=["primary", "secondary", "tertiary", "unknown"],
                    value="tertiary"
                )
                default = gr.Dropdown(
                    label="Kredi Temerrüt Durumu",
                    choices=["yes", "no"],
                    value="no"
                )
                balance = gr.Number(label="Yıllık Ortalama Bakiye (€)", value=1500, minimum=-10000, maximum=100000)
                housing = gr.Dropdown(
                    label="Konut Kredisi",
                    choices=["yes", "no"],
                    value="yes"
                )
                loan = gr.Dropdown(
                    label="Bireysel Kredi",
                    choices=["yes", "no"],
                    value="no"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Kampanya Bilgileri")
                contact = gr.Dropdown(
                    label="İletişim Türü",
                    choices=["cellular", "telephone", "unknown"],
                    value="cellular"
                )
                day = gr.Number(label="Son İletişim Günü (Ayın Kaçıncı Günü)", value=15, minimum=1, maximum=31)
                month = gr.Dropdown(
                    label="Son İletişim Ayı",
                    choices=["jan", "feb", "mar", "apr", "may", "jun",
                            "jul", "aug", "sep", "oct", "nov", "dec"],
                    value="may"
                )
                campaign = gr.Number(label="Bu Kampanyada Yapılan İletişim Sayısı", value=2, minimum=1, maximum=50)
                pdays = gr.Number(
                    label="Önceki Kampanyadan Bu Yana Geçen Gün (-1: Daha Önce İletişim Yok)",
                    value=-1,
                    minimum=-1,
                    maximum=1000
                )
                previous = gr.Number(label="Önceki Kampanyalarda Yapılan İletişim Sayısı", value=0, minimum=0, maximum=100)
                poutcome = gr.Dropdown(
                    label="Önceki Kampanya Sonucu",
                    choices=["success", "failure", "other", "unknown"],
                    value="unknown"
                )
        
        predict_btn = gr.Button("🔮 Tahmin Yap", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Sonuçlar")
                prediction = gr.Textbox(label="Tahmin", interactive=False)
                probability = gr.Number(label="Olasılık", interactive=False)
                risk_level = gr.Textbox(label="Risk Seviyesi", interactive=False)
                explanation = gr.Markdown(label="Açıklama")
        
        predict_btn.click(
            fn=predict_gradio,
            inputs=[age, job, marital, education, default, balance,
                   housing, loan, contact, day, month, campaign,
                   pdays, previous, poutcome],
            outputs=[prediction, probability, risk_level, explanation]
        )
        
        gr.Markdown("""
        ### 📊 Model Bilgileri
        - **Model Tipi:** LightGBM
        - **AUC Score:** 0.7828
        - **F1 Score:** 0.6842
        - **Precision:** 0.7534
        - **Recall:** 0.6267
        - **Production-Ready:** ✅ (duration feature yok)
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

