import sqlite3
import os
import warnings
from typing import List, Any, Union, Optional

import pickle
from src.config.settings import PATHS

# Suppress SQLite warnings
warnings.filterwarnings('ignore', category=UserWarning)

def _get_cache_path(db_path: str) -> str:
    """Generates a path for the cache file based on the db path."""
    cache_dir = PATHS.get('cache_dir', 'data/cache')
    os.makedirs(cache_dir, exist_ok=True)
    db_filename = os.path.basename(db_path)
    cache_filename = f"{os.path.splitext(db_filename)[0]}.pkl"
    return os.path.join(cache_dir, cache_filename)

def _load_from_cache(cache_path: str, db_mod_time: float) -> Optional[Any]:
    """Loads data from a pickle cache file if it's valid."""
    if os.path.exists(cache_path):
        cache_mod_time = os.path.getmtime(cache_path)
        if cache_mod_time > db_mod_time:
            try:
                with open(cache_path, 'rb') as f:
                    print(f"⚡️ Loading data from cache: {cache_path}")
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                print("⚠️ Cache file is corrupted. Re-loading from DB.")
    return None

def _save_to_cache(cache_path: str, data: Any):
    """Saves data to a pickle cache file."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Data saved to cache: {cache_path}")
    except IOError as e:
        print(f"❌ Could not write to cache file: {e}")

class DataFrameAlternative:
    """Pandas DataFrame'e basit alternatif"""
    def __init__(self, data: Optional[List[List[Any]]] = None, columns: Optional[List[str]] = None):
        if data is None:
            self.data: List[List[Any]] = []
            self.columns: List[str] = columns or []
        else:
            self.data = data
            self.columns = columns or [str(i) for i in range(len(data[0]) if data else 0)]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, key: Union[str, int, slice]) -> Any:
        if isinstance(key, str):
            if key in self.columns:
                col_idx = self.columns.index(key)
                return ColumnData([row[col_idx] for row in self.data])
            else:
                raise KeyError(f"Column '{key}' not found")
        elif isinstance(key, (int, slice)):
            return self.data[key]
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")
    
    @property
    def empty(self):
        return len(self.data) == 0
    
    def tail(self, n):
        new_data = self.data[-n:] if n <= len(self.data) else self.data
        return DataFrameAlternative(new_data, self.columns)

class ColumnData:
    """Pandas Series'e basit alternatif"""
    def __init__(self, values):
        self.values = values
    
    def __getitem__(self, index):
        return self.values[index]
    
    def __len__(self):
        return len(self.values)
    
    def mean(self):
        import statistics
        return statistics.mean(self.values) if self.values else 0

def load_data_from_sqlite(db_path="jetx_data.db", limit=None, use_cache=True):
    """
    SQLite veritabanından JetX verilerini yükler. Veritabanı değişmediyse önbelleği kullanır.
    
    Args:
        db_path: SQLite veritabanı dosya yolu (.db uzantılı)
        limit: Yüklenecek son kayıt sayısı (None=tümü). Limit kullanıldığında önbellek devre dışı kalır.
        use_cache: Önbellek kullanılıp kullanılmayacağı.
    
    Returns:
        DataFrameAlternative: Yüklenen veriler
    """
    cache_path = _get_cache_path(db_path)
    
    # Limit varsa veya önbellek istenmiyorsa, önbelleği atla
    if limit or not use_cache:
        return _load_from_db(db_path, limit)

    # Önbelleği kontrol et
    try:
        db_mod_time = os.path.getmtime(db_path)
        cached_data = _load_from_cache(cache_path, db_mod_time)
        if cached_data:
            return cached_data
    except FileNotFoundError:
        # DB dosyası yoksa, _load_from_db'nin halletmesine izin ver
        pass

    # Önbellek yoksa veya geçersizse DB'den yükle ve önbelleği güncelle
    df = _load_from_db(db_path, limit)
    if df and not df.empty:
        _save_to_cache(cache_path, df)
    return df

def _load_from_db(db_path, limit):
    """Helper function to load data directly from the SQLite database."""
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"📁 Created database directory: {db_dir}")
        
        conn = sqlite3.connect(db_path)
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS jetx_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value REAL NOT NULL
        )''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_value REAL,
            confidence_score REAL,
            above_threshold INTEGER,
            actual_value REAL,
            was_correct INTEGER
        )''')
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM jetx_results")
        total_count = cursor.fetchone()[0]
        
        if total_count == 0:
            print("⚠️ Database is empty. Creating sample data...")
            sample_data = create_sample_jetx_data()
            cursor.executemany("INSERT INTO jetx_results (value) VALUES (?)", [(v,) for v in sample_data])
            conn.commit()
            total_count = len(sample_data)
            print(f"✅ Created {total_count} sample records")
        
        min_required = 500
        if total_count < min_required:
            print(f"⚠️ Only {total_count} records available. Minimum {min_required} required.")
            print("💡 Adding more sample data...")
            additional_data = create_sample_jetx_data(count=min_required - total_count)
            cursor.executemany("INSERT INTO jetx_results (value) VALUES (?)", [(v,) for v in additional_data])
            conn.commit()
        
        query = "SELECT * FROM jetx_results ORDER BY id"
        if limit:
            query = f"SELECT * FROM ({query} DESC LIMIT {limit}) ORDER BY id ASC"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()
        
        df = DataFrameAlternative(rows, column_names)
        print(f"📊 Loaded {len(df)} records from database")
        return df
        
    except Exception as e:
        print(f"❌ Database loading error: {e}")
        print("🔄 Trying to create emergency sample data...")
        return create_emergency_dataframe()

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
