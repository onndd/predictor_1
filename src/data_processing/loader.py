import sqlite3
import os
import warnings
from typing import List, Any, Union, Optional

# Suppress SQLite warnings
warnings.filterwarnings('ignore', category=UserWarning)

def save_result_to_sqlite(value, db_path="jetx_data.db"):
    """
    Yeni bir JetX sonucunu veritabanına kaydeder
    
    Args:
        value: JetX oyun sonucu (katsayı)
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tablo yoksa oluştur
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS jetx_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value REAL NOT NULL
    )
    ''')
    
    cursor.execute('''
    INSERT INTO jetx_results (value) VALUES (?)
    ''', (value,))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    
    return record_id

def save_prediction_to_sqlite(prediction_data, db_path="jetx_data.db"):
    """
    Tahmin sonuçlarını SQLite veritabanına kaydeder
    
    Args:
        prediction_data: Kaydedilecek tahmin verisi (dict)
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Tablo yoksa oluştur
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_value REAL,
        confidence_score REAL,
        above_threshold INTEGER,
        actual_value REAL,
        was_correct INTEGER
    )
    ''')
    
    cursor.execute('''
    INSERT INTO predictions 
    (predicted_value, confidence_score, above_threshold)
    VALUES (?, ?, ?)
    ''', (
        prediction_data['predicted_value'],
        prediction_data['confidence_score'],
        1 if prediction_data['above_threshold'] else 0
    ))
    
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    
    return record_id

def update_prediction_result(prediction_id, actual_value, db_path="jetx_data.db"):
    """
    Tahmin sonucunu günceller (gerçek değer öğrenildiğinde)
    
    Args:
        prediction_id: Güncellenecek tahmin ID'si
        actual_value: Gerçekleşen JetX değeri
        db_path: SQLite veritabanı dosya yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Önce mevcut tahmini çek
    cursor.execute("SELECT above_threshold FROM predictions WHERE id=?", (prediction_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False
    
    above_threshold = row[0]
    was_correct = 1 if (above_threshold == 1 and actual_value >= 1.5) or \
                       (above_threshold == 0 and actual_value < 1.5) else 0
    
    # Tahmini güncelle
    cursor.execute('''
    UPDATE predictions 
    SET actual_value=?, was_correct=? 
    WHERE id=?
    ''', (actual_value, was_correct, prediction_id))
    
    conn.commit()
    conn.close()
    
    return True

def create_sample_jetx_data(count: int = 1000) -> List[float]:
    """
    Örnek JetX verisi oluşturur
    
    Args:
        count: Oluşturulacak veri sayısı
        
    Returns:
        List[float]: Örnek JetX değerleri
    """
    import random
    import numpy as np
    
    # Gerçekçi JetX dağılımı
    sample_data = []
    
    for _ in range(count):
        # %70 crash (1.0-1.5 arası)
        # %20 orta (1.5-3.0 arası)  
        # %10 yüksek (3.0-20.0 arası)
        rand = random.random()
        
        if rand < 0.7:
            # Crash bölgesi
            value = random.uniform(1.0, 1.5)
        elif rand < 0.9:
            # Orta bölge
            value = random.uniform(1.5, 3.0)
        else:
            # Yüksek bölge (exponential distribution)
            value = min(20.0, random.expovariate(0.5) + 3.0)
        
        sample_data.append(round(value, 2))
    
    return sample_data

def create_emergency_dataframe() -> DataFrameAlternative:
    """
    Acil durum için basit veri oluşturur
    
    Returns:
        DataFrameAlternative: Acil durum verisi
    """
    print("🚨 Creating emergency sample data...")
    
    # Basit pattern: 1.0-2.0 arası random değerler
    emergency_data = []
    for i in range(1000):
        value = 1.0 + (i % 10) * 0.1 + (i % 3) * 0.05
        emergency_data.append([i + 1, value])
    
    df = DataFrameAlternative(emergency_data, ['id', 'value'])
    print(f"✅ Created emergency dataframe with {len(df)} records")
    return df
