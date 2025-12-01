"""
Inference Module
================
Egitilmis modelden tahmin alma ve gerekli preprocessing islemleri.

Bu modul, egitilmis modeli yukleyip, yeni veriler uzerinde tahmin yapmak icin
gerekli tum preprocessing ve feature engineering islemlerini gerceklestirir.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
from sklearn.preprocessing import LabelEncoder
import logging

from config import FINAL_MODEL_DIR, PROCESSED_DATA_DIR

# Logging ayarlari
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.label_encoders = None
        self.feature_engineering_func = None
        
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
        
        # Model ve metadata'yi ayir
        if isinstance(model_data, dict):
            self.model = model_data.get("model")
            self.preprocessor = model_data.get("preprocessor")
            self.feature_names = model_data.get("feature_names")
            self.label_encoders = model_data.get("label_encoders")
            self.feature_engineering_func = model_data.get("feature_engineering_func")
        else:
            self.model = model_data
            
        logger.info(f"Model basariyla yuklendi: {model_path}")
        if self.label_encoders:
            logger.info(f"Label encoder sayisi: {len(self.label_encoders)}")
        if self.feature_names:
            logger.info(f"Feature sayisi: {len(self.feature_names)}")
        
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
            logger.info(f"Preprocessor basariyla yuklendi: {preprocessor_path}")
        else:
            logger.warning("Preprocessor dosyasi bulunamadi, raw data ile devam edilecek.")
    
    def _apply_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering pipeline'ını uygular.
        
        Args:
            X: Ham veri DataFrame
            
        Returns:
            Feature engineering uygulanmış DataFrame
        """
        if self.label_encoders is None:
            raise ValueError("Label encoder'lar yuklenmedi. Once load_model() cagirin.")
        
        X_fe = X.copy()
        
        # Duration'ı çıkar (production'da bilinmez)
        if 'duration' in X_fe.columns:
            X_fe = X_fe.drop(columns=['duration'])
        
        # 1. Yaş grupları
        X_fe['age_group'] = pd.cut(
            X_fe['age'], 
            bins=[0, 30, 40, 50, 60, 100], 
            labels=[0, 1, 2, 3, 4]
        )
        
        # 2. Bakiye kategorileri
        X_fe['balance_category'] = pd.cut(
            X_fe['balance'], 
            bins=[-np.inf, 0, 100, 500, 2000, np.inf], 
            labels=[0, 1, 2, 3, 4]
        )
        
        # 3. Never contacted flag
        X_fe['never_contacted'] = (X_fe['pdays'] == -1).astype(int)
        
        # 4. Mevsimsellik
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        X_fe['month_num'] = X_fe['month'].map(month_map)
        X_fe['is_quarter_end'] = X_fe['month_num'].isin([3, 6, 9, 12]).astype(int)
        
        # 5. Kampanya metrikleri
        X_fe['total_contacts'] = X_fe['campaign'] + X_fe['previous']
        X_fe['over_contacted'] = (X_fe['campaign'] > 5).astype(int)
        
        # 6. İnteraksiyon feature'ları
        X_fe['age_balance_interaction'] = X_fe['age'] * (X_fe['balance'] / 1000)
        X_fe['age_campaign_interaction'] = X_fe['age'] * X_fe['campaign']
        X_fe['balance_per_age'] = X_fe['balance'] / (X_fe['age'] + 1)
        
        # Sonsuz değerleri kontrol et ve düzelt
        for col in X_fe.select_dtypes(include=[np.number]).columns:
            if np.isinf(X_fe[col]).any():
                X_fe[col] = X_fe[col].replace([np.inf, -np.inf], np.nan)
                X_fe[col] = X_fe[col].fillna(X_fe[col].median())
        
        # Label Encoding
        cat_cols = X_fe.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Görülmemiş kategorileri handle et
                X_fe[col] = X_fe[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        return X_fe
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Veriyi on isle (feature engineering dahil).
        
        Args:
            data: Ham veri DataFrame'i
            
        Returns:
            Islenmis veri DataFrame'i
        """
        # Önce feature engineering uygula
        if self.label_encoders is not None:
            processed_data = self._apply_feature_engineering(data)
        elif self.preprocessor is not None:
            # Eski preprocessor varsa onu kullan
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
        else:
            # Preprocessor yoksa direkt dondur
            processed_data = data
        
        # Feature sırasını kontrol et
        if self.feature_names is not None:
            # Eksik feature'ları 0 ile doldur
            missing_features = set(self.feature_names) - set(processed_data.columns)
            for feat in missing_features:
                processed_data[feat] = 0
            
            # Feature sırasını düzelt
            processed_data = processed_data[self.feature_names]
        
        return processed_data
    
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
        
        Business kurallari:
        - Dusuk Risk (< 0.3): Arama yapilmamali
        - Orta Risk (0.3-0.5): Normal oncelik
        - Yuksek Risk (0.5-0.7): Yuksek oncelik
        - Cok Yuksek Risk (>= 0.7): Oncelikli arama
        
        Args:
            probability: Tahmin olasiligi (0-1 arasi)
            
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
    
    def get_model_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Model metriklerini getir (eger model metadata'sinda varsa).
        
        Returns:
            Model metrikleri dictionary veya None
        """
        # Bu metod model metadata'sindan metrikleri cekmek icin kullanilabilir
        # Simdilik None donuyor, ileride genisletilebilir
        return None


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
    
    # Ornek tahmin (Bank Marketing dataset feature'ları)
    sample_features = {
        "age": 45,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
    
    result = inference.predict_single(sample_features)
    print(f"Tahmin: {result}")


if __name__ == "__main__":
    main()

