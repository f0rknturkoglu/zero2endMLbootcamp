"""
Inference Module
================
Egitilmis modelden tahmin alma ve gerekli preprocessing islemleri.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List

from config import FINAL_MODEL_DIR, PROCESSED_DATA_DIR


class ModelInference:
    """
    Egitilmis model ile tahmin yapma sinifi.
    
    Kullanim:
        inference = ModelInference()
        inference.load_model("final_model.pkl")
        predictions = inference.predict(data)
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
    def load_model(self, model_path: Union[str, Path] = None):
        """
        Modeli yukle.
        
        Args:
            model_path: Model dosyasinin yolu. None ise default path kullanilir.
        """
        if model_path is None:
            model_path = FINAL_MODEL_DIR / "model.pkl"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_path}")
        
        model_data = joblib.load(model_path)
        
        # Model ve preprocessor'u ayir
        if isinstance(model_data, dict):
            self.model = model_data.get("model")
            self.preprocessor = model_data.get("preprocessor")
            self.feature_names = model_data.get("feature_names")
        else:
            self.model = model_data
            
        print(f"Model basariyla yuklendi: {model_path}")
        
    def load_preprocessor(self, preprocessor_path: Union[str, Path] = None):
        """
        Preprocessor'u yukle.
        
        Args:
            preprocessor_path: Preprocessor dosyasinin yolu.
        """
        if preprocessor_path is None:
            preprocessor_path = FINAL_MODEL_DIR / "preprocessor.pkl"
            
        preprocessor_path = Path(preprocessor_path)
        
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            print(f"Preprocessor basariyla yuklendi: {preprocessor_path}")
        else:
            print("Preprocessor dosyasi bulunamadi, raw data ile devam edilecek.")
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Veriyi on isle.
        
        Args:
            data: Ham veri DataFrame'i
            
        Returns:
            Islenmis veri DataFrame'i
        """
        if self.preprocessor is not None:
            # Preprocessor varsa uygula
            processed_data = self.preprocessor.transform(data)
            
            # Eger numpy array donerse DataFrame'e cevir
            if isinstance(processed_data, np.ndarray):
                if self.feature_names is not None:
                    processed_data = pd.DataFrame(
                        processed_data, 
                        columns=self.feature_names
                    )
                else:
                    processed_data = pd.DataFrame(processed_data)
                    
            return processed_data
        else:
            # Preprocessor yoksa direkt dondur
            return data
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Tahmin yap.
        
        Args:
            data: Tahmin yapilacak veri
            
        Returns:
            Tahminler (0/1)
        """
        if self.model is None:
            raise ValueError("Model yuklenmedi. Once load_model() cagirin.")
        
        # Preprocess
        processed_data = self.preprocess(data)
        
        # Tahmin
        predictions = self.model.predict(processed_data)
        
        return predictions
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Olasilik tahmini yap.
        
        Args:
            data: Tahmin yapilacak veri
            
        Returns:
            Olasiliklar (0-1 arasi)
        """
        if self.model is None:
            raise ValueError("Model yuklenmedi. Once load_model() cagirin.")
        
        # Preprocess
        processed_data = self.preprocess(data)
        
        # Olasilik tahmini
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(processed_data)[:, 1]
        else:
            # Bazi modeller predict_proba desteklemez
            probabilities = self.model.predict(processed_data)
        
        return probabilities
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tek bir ornek icin tahmin yap.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tahmin sonucu dictionary
        """
        # Dictionary'i DataFrame'e cevir
        data = pd.DataFrame([features])
        
        # Tahmin
        prediction = self.predict(data)[0]
        probability = self.predict_proba(data)[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": self._get_risk_level(probability)
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Olasiliga gore risk seviyesi belirle.
        
        Args:
            probability: Tahmin olasiligi
            
        Returns:
            Risk seviyesi string
        """
        if probability < 0.3:
            return "Dusuk Risk"
        elif probability < 0.5:
            return "Orta Risk"
        elif probability < 0.7:
            return "Yuksek Risk"
        else:
            return "Cok Yuksek Risk"


def main():
    """Test fonksiyonu."""
    # Ornek kullanim
    inference = ModelInference()
    
    # Model yukle
    try:
        inference.load_model()
    except FileNotFoundError as e:
        print(f"Hata: {e}")
        print("Once modeli egitmeniz gerekiyor.")
        return
    
    # Ornek tahmin (kendi feature'larinizi yazin)
    sample_features = {
        "feature1": 1.0,
        "feature2": 2.0,
        # ...
    }
    
    result = inference.predict_single(sample_features)
    print(f"Tahmin: {result}")


if __name__ == "__main__":
    main()

