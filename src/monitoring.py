import sqlite3
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

class MonitoringDB:
    def __init__(self, db_path="monitoring.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Veritabanini ve tabloyu olustur"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Logs tablosu
        c.execute('''
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                age INTEGER,
                job TEXT,
                balance REAL,
                campaign INTEGER,
                prediction INTEGER,
                probability REAL,
                risk_level TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_prediction(self, features, result):
        """Tahmini logla"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            c.execute('''
                INSERT INTO prediction_logs 
                (timestamp, age, job, balance, campaign, prediction, probability, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                features.get('age'),
                features.get('job'),
                features.get('balance'),
                features.get('campaign'),
                result.get('prediction'),
                result.get('probability'),
                result.get('risk_level')
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Loglama hatasi: {e}")

    def get_recent_logs(self, limit=10):
        """Son loglari getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM prediction_logs ORDER BY id DESC LIMIT {limit}"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()

    def get_stats(self):
        """Istatistikleri getir"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Toplam tahmin sayisi
            total_count = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
            
            # Pozitif tahmin orani
            positive_count = conn.execute("SELECT COUNT(*) FROM prediction_logs WHERE prediction = 1").fetchone()[0]
            
            # Ortalama olasilik
            avg_prob = conn.execute("SELECT AVG(probability) FROM prediction_logs").fetchone()[0] or 0
            
            conn.close()
            
            return {
                "total_predictions": total_count,
                "positive_rate": (positive_count / total_count * 100) if total_count > 0 else 0,
                "avg_probability": avg_prob
            }
        except Exception:
            return {"total_predictions": 0, "positive_rate": 0, "avg_probability": 0}
