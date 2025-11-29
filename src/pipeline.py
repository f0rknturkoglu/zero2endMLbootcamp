"""
ML Pipeline
===========
Tum ML akisinin gerceklestigi final pipeline scripti.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)

import lightgbm as lgb

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FINAL_MODEL_DIR,
    TRAIN_DATA,
    TARGET_COLUMN,
    ID_COLUMN,
    RANDOM_STATE,
    CV_FOLDS,
    TEST_SIZE,
    LGBM_DEFAULT_PARAMS
)


class MLPipeline:
    """
    End-to-end ML Pipeline sinifi.
    
    Bu sinif veri yukleme, on isleme, model egitimi ve 
    degerlendirme adimlarini icerir.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Pipeline'i baslat.
        
        Args:
            config: Ozel konfigurasyon dictionary'si
        """
        self.config = config or {}
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.metrics = {}
        
    def load_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Veriyi yukle.
        
        Args:
            data_path: Veri dosyasinin yolu
            
        Returns:
            Yuklenen DataFrame
        """
        if data_path is None:
            data_path = TRAIN_DATA
            
        print(f"Veri yukleniyor: {data_path}")
        
        # Dosya uzantisina gore yukle
        if str(data_path).endswith('.csv'):
            df = pd.read_csv(data_path)
        elif str(data_path).endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif str(data_path).endswith('.xlsx'):
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"Desteklenmeyen dosya formati: {data_path}")
        
        print(f"Veri yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")
        
        return df
    
    def preprocess(self, df: pd.DataFrame, is_train: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Veriyi on isle.
        
        Args:
            df: Ham veri
            is_train: Egitim verisi mi?
            
        Returns:
            Islenmis features ve target (eger varsa)
        """
        print("On isleme baslaniyor...")
        
        # Target'i ayir
        if is_train and TARGET_COLUMN in df.columns:
            y = df[TARGET_COLUMN]
            X = df.drop(columns=[TARGET_COLUMN])
        else:
            y = None
            X = df.copy()
        
        # ID sutununu cikar
        if ID_COLUMN in X.columns:
            X = X.drop(columns=[ID_COLUMN])
        
        # Kategorik ve numerik sutunlari belirle
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Kategorik sutunlar: {len(categorical_cols)}")
        print(f"Numerik sutunlar: {len(numeric_cols)}")
        
        # Eksik degerleri doldur
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
            
        for col in categorical_cols:
            X[col] = X[col].fillna("MISSING")
        
        # Label encoding (kategorik icin)
        if is_train:
            self.label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        else:
            for col in categorical_cols:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Gorulmemis kategorileri handle et
                    X[col] = X[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        self.feature_names = X.columns.tolist()
        
        print(f"On isleme tamamlandi: {X.shape[0]} satir, {X.shape[1]} feature")
        
        return X, y
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Modeli egit.
        
        Args:
            X: Feature matrix
            y: Target vector
            params: Model parametreleri
        """
        print("Model egitimi baslaniyor...")
        
        if params is None:
            params = LGBM_DEFAULT_PARAMS.copy()
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]} ornek")
        print(f"Validation set: {X_val.shape[0]} ornek")
        
        # LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Model egitimi
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Validation metrikleri
        y_pred_proba = self.model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        self.metrics['val_auc'] = roc_auc_score(y_val, y_pred_proba)
        self.metrics['val_accuracy'] = accuracy_score(y_val, y_pred)
        self.metrics['val_precision'] = precision_score(y_val, y_pred)
        self.metrics['val_recall'] = recall_score(y_val, y_pred)
        self.metrics['val_f1'] = f1_score(y_val, y_pred)
        
        print("\n=== Validation Metrikleri ===")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_folds: int = CV_FOLDS
    ) -> Dict[str, float]:
        """
        Cross-validation yap.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_folds: Fold sayisi
            
        Returns:
            CV metrikleri
        """
        print(f"\n{n_folds}-Fold Cross Validation baslaniyor...")
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # LightGBM dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Model egitimi
            model = lgb.train(
                LGBM_DEFAULT_PARAMS,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)  # Sessiz mod
                ]
            )
            
            # Tahmin
            y_pred_proba = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred_proba)
            auc_scores.append(auc)
            
            print(f"Fold {fold + 1}: AUC = {auc:.4f}")
        
        cv_results = {
            'cv_auc_mean': np.mean(auc_scores),
            'cv_auc_std': np.std(auc_scores),
            'cv_auc_scores': auc_scores
        }
        
        print(f"\nCV AUC: {cv_results['cv_auc_mean']:.4f} (+/- {cv_results['cv_auc_std']:.4f})")
        
        return cv_results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Feature importance'lari getir.
        
        Args:
            top_n: Gosterilecek feature sayisi
            
        Returns:
            Feature importance DataFrame
        """
        if self.model is None:
            raise ValueError("Model henuz egitilmedi.")
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        })
        
        importance = importance.sort_values('importance', ascending=False)
        
        return importance.head(top_n)
    
    def save_model(self, model_name: Optional[str] = None) -> Path:
        """
        Modeli kaydet.
        
        Args:
            model_name: Model dosya adi
            
        Returns:
            Kaydedilen dosya yolu
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok.")
        
        # Klasoru olustur
        FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}.pkl"
        
        model_path = FINAL_MODEL_DIR / model_name
        
        # Model ve metadata'yi kaydet
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'label_encoders': getattr(self, 'label_encoders', None),
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        
        print(f"Model kaydedildi: {model_path}")
        
        return model_path
    
    def run(self, data_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Tum pipeline'i calistir.
        
        Args:
            data_path: Veri dosyasi yolu
            
        Returns:
            Pipeline sonuclari
        """
        print("=" * 60)
        print("ML PIPELINE BASLADI")
        print("=" * 60)
        
        # 1. Veri yukle
        df = self.load_data(data_path)
        
        # 2. On isleme
        X, y = self.preprocess(df, is_train=True)
        
        # 3. Cross-validation
        cv_results = self.cross_validate(X, y)
        
        # 4. Final model egitimi
        self.train(X, y)
        
        # 5. Feature importance
        importance = self.get_feature_importance()
        print("\n=== Top 10 Feature Importance ===")
        print(importance.head(10).to_string(index=False))
        
        # 6. Model kaydet
        model_path = self.save_model("model.pkl")
        
        print("\n" + "=" * 60)
        print("ML PIPELINE TAMAMLANDI")
        print("=" * 60)
        
        return {
            'metrics': self.metrics,
            'cv_results': cv_results,
            'feature_importance': importance,
            'model_path': model_path
        }


def main():
    """Ana fonksiyon."""
    pipeline = MLPipeline()
    
    try:
        results = pipeline.run()
    except FileNotFoundError as e:
        print(f"\nHata: {e}")
        print("\nOnce data klasorune veri dosyanizi eklemeniz gerekiyor.")
        print("Kaggle'dan veri indirip data/raw/ klasorune koyun.")
        print("Ardindan config.py dosyasinda TRAIN_DATA path'ini guncelleyin.")


if __name__ == "__main__":
    main()

