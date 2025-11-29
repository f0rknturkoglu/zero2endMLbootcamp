"""
Web Application
===============
REST API (FastAPI) ve arayuz (Gradio/Streamlit).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
import gradio as gr
from pathlib import Path
import pandas as pd

from config import API_HOST, API_PORT, FINAL_MODEL_DIR
from inference import ModelInference

# ============================================================================
# FASTAPI UYGULAMASI
# ============================================================================

app = FastAPI(
    title="ML Model API",
    description="Makine ogrenmesi modeli tahmin API'si",
    version="1.0.0"
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
    features: Dict[str, Any]


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
        print("Model basariyla yuklendi!")
    except FileNotFoundError:
        print("UYARI: Model dosyasi bulunamadi. /predict endpoint'i calismayacak.")


@app.get("/")
async def root():
    """Ana sayfa."""
    return {
        "message": "ML Model API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Saglik kontrolu."""
    return {
        "status": "healthy",
        "model_loaded": inference.model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Tek bir ornek icin tahmin yap.
    
    Args:
        request: Feature dictionary iceren istek
        
    Returns:
        Tahmin sonucu
    """
    if inference.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model yuklenmedi. Lutfen modeli egitip kaydedin."
        )
    
    try:
        result = inference.predict_single(request.features)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Birden fazla ornek icin tahmin yap.
    
    Args:
        request: Feature dictionary listesi iceren istek
        
    Returns:
        Tahmin sonuclari listesi
    """
    if inference.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model yuklenmedi. Lutfen modeli egitip kaydedin."
        )
    
    try:
        predictions = []
        for features in request.data:
            result = inference.predict_single(features)
            predictions.append(PredictionResponse(**result))
        
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Model bilgilerini getir."""
    if inference.model is None:
        return {"error": "Model yuklenmedi"}
    
    return {
        "feature_names": inference.feature_names,
        "model_type": type(inference.model).__name__
    }


# ============================================================================
# GRADIO ARAYUZU
# ============================================================================

def create_gradio_interface():
    """Gradio arayuzu olustur."""
    
    def predict_gradio(*args):
        """Gradio tahmin fonksiyonu."""
        if inference.model is None:
            return "Model yuklenmedi!", 0.0, "Bilinmiyor"
        
        # Feature isimlerini al
        feature_names = inference.feature_names or [f"feature_{i}" for i in range(len(args))]
        
        # Features dictionary olustur
        features = dict(zip(feature_names[:len(args)], args))
        
        try:
            result = inference.predict_single(features)
            return (
                f"Tahmin: {'Pozitif' if result['prediction'] == 1 else 'Negatif'}",
                result['probability'],
                result['risk_level']
            )
        except Exception as e:
            return f"Hata: {str(e)}", 0.0, "Hata"
    
    # Ornek input'lar (kendi feature'lariniza gore guncelleyin)
    inputs = [
        gr.Number(label="Feature 1", value=0),
        gr.Number(label="Feature 2", value=0),
        gr.Number(label="Feature 3", value=0),
        # Daha fazla feature ekleyin...
    ]
    
    outputs = [
        gr.Textbox(label="Tahmin"),
        gr.Number(label="Olasilik"),
        gr.Textbox(label="Risk Seviyesi")
    ]
    
    interface = gr.Interface(
        fn=predict_gradio,
        inputs=inputs,
        outputs=outputs,
        title="ML Model Tahmin Arayuzu",
        description="Makine ogrenmesi modeli ile tahmin yapin.",
        theme="default"
    )
    
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
    except FileNotFoundError:
        print("UYARI: Model dosyasi bulunamadi.")
    
    interface = create_gradio_interface()
    interface.launch(share=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "gradio":
        run_gradio()
    else:
        run_api()

